"""Thử nghiệm tìm kiếm bài viết theo từ khóa trong nhóm Facebook."""
import json
import sys
import time
import urllib.request
from urllib.parse import quote
import websocket

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")


def search_group(keyword, group_slug="2k5ptit"):
    tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
    tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=15, suppress_origin=True)

    url = f"https://www.facebook.com/groups/{group_slug}/search/?q={quote(keyword)}"
    print(f"Đang tìm kiếm từ khóa: {keyword} -> {url}")
    ws.send(json.dumps({"id": 1, "method": "Page.navigate", "params": {"url": url}}))
    time.sleep(4.0)

    # Cuộn 3 lần
    for _ in range(3):
        ws.send(json.dumps({"id": 2, "method": "Runtime.evaluate", "params": {"expression": "window.scrollBy(0, 1500);"}}))
        time.sleep(1.2)

    ws.send(json.dumps({
        "id": 3,
        "method": "Runtime.evaluate",
        "params": {
            "expression": """
            (() => {
                const anchors = Array.from(document.querySelectorAll('a[href*="/posts/"], a[href*="/permalink/"]'));
                const posts = [];
                for (const a of anchors) {
                    const m = a.href.match(/posts\\/(\\d+)/) || a.href.match(/permalink\\/(\\d+)/);
                    if (m) posts.push(m[1]);
                }
                return Array.from(new Set(posts));
            })()
            """,
            "returnByValue": True
        }
    }))
    res = json.loads(ws.recv())
    posts = res.get("result", {}).get("result", {}).get("value", [])
    print(f"Từ khóa '{keyword}' tìm thấy: {len(posts)} bài viết:")
    for p in posts[:10]:
        print(f"  - https://www.facebook.com/groups/{group_slug}/posts/{p}/")

    ws.close()
    return posts


if __name__ == "__main__":
    search_group("review")
