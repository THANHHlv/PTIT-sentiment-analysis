"""Kiểm tra HTML công khai; không đăng nhập, không gọi API Facebook nội bộ."""
import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from ptit_sentiment.common import load_config, run_cli, write_json


class VisibleSummary(HTMLParser):
    """Đếm liên kết bài và văn bản ngoài script, không lưu thông tin tác giả."""
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.text_parts = 0
        self.post_links = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag == "a":
            href = dict(attrs).get("href", "")
            if "/posts/" in href or "/permalink/" in href:
                self.post_links += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.text_parts += 1


class JSONScripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.buffer = []
        self.documents = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            values = dict(attrs)
            self.active = values.get("type") == "application/json"
            self.buffer = []

    def handle_data(self, data):
        if self.active:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.active:
            try:
                self.documents.append(json.loads("".join(self.buffer)))
            except json.JSONDecodeError:
                pass
            self.active = False


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def embedded_documents(html):
    """Đọc JSON trong bootstrap công khai, không thực thi JavaScript hoặc gọi endpoint nội bộ."""
    parser = JSONScripts()
    parser.feed(html)
    yield from parser.documents
    decoder = json.JSONDecoder()
    # JS bootstrap có thể bọc JSON trong lời gọi hàm; chỉ decode object JSON hợp lệ.
    for match in re.finditer(r'\{"(?:__bbox|require)"\s*:', html):
        try:
            value, _ = decoder.raw_decode(html[match.start():])
        except json.JSONDecodeError:
            continue
        yield value


def probe(url, timeout=20, retries=1):
    """Đọc tối đa 8 MiB HTML; retry lỗi mạng/5xx, dừng ở 403/429 và login."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.facebook.com" or not parsed.path.startswith("/groups/"):
        raise ValueError("Chỉ kiểm tra URL https://www.facebook.com/groups/... do người dùng chỉ định.")
    info = {"url": url, "checked_at": datetime.now(timezone.utc).isoformat(), "attempts": 0}
    for attempt in range(retries + 1):
        info["attempts"] += 1
        try:
            with urlopen(Request(url), timeout=timeout) as response:
                body = response.read(8 * 1024 * 1024 + 1)
                if len(body) > 8 * 1024 * 1024:
                    raise ValueError("HTML vượt giới hạn 8 MiB.")
                info.update(http_status=response.status, final_url=response.url, bytes=len(body))
            html = body.decode("utf-8", errors="replace")
            match = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
            info["title"] = unescape(match.group(1)) if match else None
            visible = VisibleSummary()
            visible.feed(html)
            info["visible_text_parts"] = visible.text_parts
            info["public_post_links"] = visible.post_links
            typed = Counter()
            comment_shapes = []
            for document in embedded_documents(html):
                for item in objects(document):
                    typename = item.get("__typename")
                    if isinstance(typename, str):
                        typed[typename] += 1
                        if typename == "Comment":
                            comment_shapes.append(sorted(item.keys()))
            info["rendered_json_types"] = dict(typed)
            info["comment_object_shapes"] = comment_shapes[:5]
            info["comment_objects_in_initial_html"] = typed["Comment"]
            info["login_redirect"] = "/login" in info["final_url"]
            info["status"] = "html_checked"
            info["comments_collected"] = 0
            info["note"] = "HTTP 200 chỉ xác nhận trang tải được; chưa có bộ thu bình luận/phân trang đã xác minh."
            return info
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            code = getattr(error, "code", None)
            info.update(status="access_error", error=str(error), http_status=code)
            if code in (401, 403, 429) or attempt == retries:
                return info
            time.sleep(1)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/collection.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = [probe(source["source_url"]) for source in load_config(args.config)["sources"]]
    write_json(args.output, {"sources": results, "real_comments_collected": 0})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_cli(main)