"""Thu thập bình luận PTIT tự động tốc độ cao qua Chrome CDP.

Chiến lược tối ưu hóa:
1. Tìm kiếm theo khoảng thời gian (Năm 2024, 2023, 2025, 2022) kết hợp từ khóa sinh viên PTIT.
2. Tránh hoàn toàn việc lướt lại từ đầu nguồn feed hiện tại (vốn chỉ chứa các bài đã cào).
3. Hỗ trợ trích xuất cả bình luận chính (comment_id) lẫn câu trả lời (reply_comment_id).
4. Tự động đồng bộ liên tục vào data/raw/ptit_sources_v1/comments.csv.
"""
import argparse
import base64
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
import websocket

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.sync_browser_capture import sync

GROUPS = {
    "ptit_2k5": {
        "slug": "450310163844833",
        "name": "2k5ptit",
        "url": "https://www.facebook.com/groups/450310163844833",
        "prefix": "ptit_2k5"
    },
    "ptit_group_584397217391365": {
        "slug": "584397217391365",
        "name": "ptit_info",
        "url": "https://www.facebook.com/groups/584397217391365",
        "prefix": "ptit_info"
    }
}

DEFAULT_TOPICS = [
    "thi", "học", "thầy", "cô", "môn", "ktx", "học phí",
    "review", "tín chỉ", "điểm", "giảng viên", "đề", "lịch thi",
    "toán", "triết", "code", "tiếng anh", "bảo lưu",
    "học lại", "cải thiện", "học bổng", "thực tập", "đồ án",
    "chuẩn đầu ra", "tốt nghiệp", "phòng trọ"
]

DEFAULT_YEARS = [2024, 2023, 2025, 2022, 2021]

def make_year_filter(year: int) -> str:
    """Tạo bộ lọc base64 ngày đăng của Facebook theo năm."""
    inner = json.dumps({
        "start_year": str(year),
        "start_month": f"{year}-1",
        "end_year": str(year),
        "end_month": f"{year}-12",
        "start_day": f"{year}-1-1",
        "end_day": f"{year}-12-31"
    })
    middle = json.dumps({"name": "creation_time", "args": inner})
    return base64.b64encode(json.dumps({"rp_creation_time:0": middle}).encode()).decode()


EXTRACT_JS = """
(() => {
    // 0. Thử chuyển bộ lọc sang "Tất cả bình luận" nếu có
    const filterBtn = Array.from(document.querySelectorAll('div[role="button"], span')).find(el => el.innerText && (el.innerText.trim().startsWith('Phù hợp nhất') || el.innerText.trim().startsWith('Bình luận hàng đầu')));
    if (filterBtn) {
        try {
            filterBtn.click();
            setTimeout(() => {
                const opt = Array.from(document.querySelectorAll('div[role="menuitem"], div[role="button"], span')).find(el => el.innerText && el.innerText.includes('Tất cả bình luận'));
                if (opt) opt.click();
            }, 300);
        } catch(e) {}
    }

    // 1. Click các nút mở rộng bình luận
    const buttons = Array.from(document.querySelectorAll('div[role="button"], span[role="button"], span'));
    for (const b of buttons) {
        const t = b.innerText ? b.innerText.trim() : '';
        const isExpand = t.includes('Xem thêm bình luận') || 
                         t.includes('Xem các bình luận trước') || 
                         t.includes('bình luận trước') || 
                         t.includes('câu trả lời trước') ||
                         t.includes('Xem thêm câu trả lời') || 
                         Boolean(t.match(/^Xem \\d+ câu trả lời/)) ||
                         Boolean(t.match(/^Xem tất cả \\d+ bình luận/));
        if (isExpand && !b.getAttribute('aria-label')?.includes('Đóng') && !b.getAttribute('aria-label')?.includes('Close')) {
            try { b.click(); } catch(e) {}
        }
    }

    // 2. Click "Xem thêm" text dài
    const seeMores = Array.from(document.querySelectorAll('div[dir="auto"] span, span[dir="auto"]'));
    for (const s of seeMores) {
        if (s.innerText && s.innerText.trim() === 'Xem thêm') {
            try { s.click(); } catch(e) {}
        }
    }

    // 3. Trích xuất bình luận (bao gồm cả comment chính và reply)
    const comments = [];
    const seenIds = new Set();
    const links = Array.from(document.querySelectorAll('a[href*="comment_id="]'));

    // Lấy postId từ URL
    const urlMatch = window.location.href.match(/(?:posts|permalink)\\/(\\d+)/);
    const postId = urlMatch ? urlMatch[1] : '';

    for (const a of links) {
        // Ưu tiên reply_comment_id nếu là câu trả lời con
        const replyMatch = a.href.match(/reply_comment_id=(\\d+)/);
        const cidMatch = a.href.match(/comment_id=(\\d+)/);
        const commentId = replyMatch ? replyMatch[1] : (cidMatch ? cidMatch[1] : null);

        if (!commentId || seenIds.has(commentId)) continue;

        // Container chứa nội dung bình luận
        let cur = a;
        let container = null;
        for (let i = 0; i < 9; i++) {
            if (!cur.parentElement) break;
            cur = cur.parentElement;
            if (cur.innerText && cur.innerText.split('\\n').length >= 3) {
                container = cur;
            }
        }
        if (!container) continue;

        const textDivs = Array.from(container.querySelectorAll('div[dir="auto"]'));
        let text = '';
        for (const div of textDivs) {
            const t = div.innerText.trim();
            if (!t) continue;
            if (t === 'Thích' || t === 'Trả lời' || t === 'Chia sẻ' || t.includes('phút') || t.includes('giờ') || t.includes('ngày') || t.includes('tuần')) continue;
            if (div.querySelector('a') && div.querySelector('a').innerText === t) continue;
            text = t;
            break;
        }

        // Bỏ '… Xem thêm' nếu còn
        text = text.replace(/[…\\.]+\\s*Xem thêm$/g, '').trim();

        if (text && text.length > 0 && commentId.match(/^\\d+$/) && postId.match(/^\\d+$/)) {
            seenIds.add(commentId);
            comments.push({
                id: commentId,
                post_id: postId,
                source_url: window.location.href.split('?')[0],
                comment_url: a.href,
                text: text,
                is_truncated: false
            });
        }
    }

    return comments;
})()
"""


class CDPClient:
    def __init__(self, wait_timeout=7200):
        self.wait_timeout = wait_timeout
        self.ws = None
        self.tab = None
        self.msg_id = 0

    def connect(self):
        start = time.time()
        first_warn = True
        while time.time() - start < self.wait_timeout:
            try:
                resp = urllib.request.urlopen("http://127.0.0.1:9222/json/list", timeout=3)
                tabs = json.loads(resp.read().decode())
                tab = next((t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", "")), None)
                if not tab:
                    tab = next((t for t in tabs if t.get("type") == "page"), None)
                if tab and "webSocketDebuggerUrl" in tab:
                    time.sleep(1.5)
                    self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=20, suppress_origin=True)
                    self.tab = tab
                    print("✅ Đã kết nối thành công Chrome cổng 9222!", flush=True)
                    return self
            except Exception:
                if first_warn:
                    print("⏳ Đang chờ kết nối Chrome cổng 9222... Hãy mở Chrome bằng: tools\\start_chrome.bat", flush=True)
                    first_warn = False
                time.sleep(2.0)
        raise ConnectionError("Không thể kết nối Chrome qua cổng 9222.")

    def reconnect(self):
        print("🔄 Đang thử kết nối lại với Chrome...", flush=True)
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        time.sleep(2.0)
        for _ in range(15):
            try:
                resp = urllib.request.urlopen("http://127.0.0.1:9222/json/list", timeout=3)
                tabs = json.loads(resp.read().decode())
                tab = next((t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", "")), None)
                if not tab:
                    tab = next((t for t in tabs if t.get("type") == "page"), None)
                if tab and "webSocketDebuggerUrl" in tab:
                    self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=20, suppress_origin=True)
                    self.tab = tab
                    print("✅ Đã kết nối lại thành công với Chrome!", flush=True)
                    return True
            except Exception:
                time.sleep(2.0)
        return False

    def send_cmd(self, method, params=None, retries=3):
        if params is None:
            params = {}
        for attempt in range(retries):
            if self.ws is None:
                if not self.reconnect():
                    time.sleep(1.0)
                    continue
            self.msg_id += 1
            cur_id = self.msg_id
            payload = json.dumps({"id": cur_id, "method": method, "params": params})
            try:
                self.ws.send(payload)
                start_recv = time.time()
                while time.time() - start_recv < 25:
                    res = json.loads(self.ws.recv())
                    if res.get("id") == cur_id:
                        return res
            except Exception as ex:
                print(f"⚠️ CDP {method} tạm gián đoạn ({type(ex).__name__}). Đang thử kết nối lại ({attempt + 1}/{retries})...", flush=True)
                self.reconnect()
        return {}

    def eval_js(self, expr):
        res = self.send_cmd("Runtime.evaluate", {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": True
        })
        return res.get("result", {}).get("result", {}).get("value")

    def navigate(self, url):
        self.send_cmd("Page.navigate", {"url": url})
        time.sleep(3.5)

    def close(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None


def get_next_snapshot_idx(capture_dir, prefix):
    existing = list(Path(capture_dir).glob(f"{prefix}_snapshot*.json"))
    max_idx = 0
    for p in existing:
        m = re.search(r"snapshot(\d+)\.json", p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def get_existing_comment_ids():
    csv_path = Path("data/raw/ptit_sources_v1/comments.csv")
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8") as f:
        return set(r["id"] for r in csv.DictReader(f))


def get_existing_post_ids():
    csv_path = Path("data/raw/ptit_sources_v1/comments.csv")
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8") as f:
        return set(r["post_id"] for r in csv.DictReader(f))


def search_posts_by_year_and_topic(client, group_slug, year, topic, existing_posts, max_scrolls=3):
    """Tìm bài viết theo năm và từ khóa trong nhóm Facebook."""
    filter_b64 = make_year_filter(year)
    search_url = f"https://www.facebook.com/groups/{group_slug}/search/?q={urllib.parse.quote(topic)}&filters={filter_b64}"
    client.navigate(search_url)
    time.sleep(3.0)

    for _ in range(max_scrolls):
        client.eval_js("window.scrollBy(0, 1500);")
        time.sleep(1.2)

    html = client.eval_js("document.body.innerHTML") or ""
    matches = set(re.findall(r'(?:post_id|story_fbid|top_level_post_id)[\'"]?:[\'"]?(\d+)[\'"]?', html))
    # Loại bỏ ID chính nhóm
    matches.discard("450310163844833")
    matches.discard("584397217391365")

    new_posts = [p for p in matches if p not in existing_posts]
    return new_posts


def scrape_post_comments(client, group_slug, prefix, post_id, existing_comment_ids, capture_dir, snapshot_idx):
    """Mở bài viết, mở rộng các tầng thảo luận và trích xuất bình luận."""
    post_url = f"https://www.facebook.com/groups/{group_slug}/posts/{post_id}/"
    client.navigate(post_url)
    time.sleep(2.5)

    # 1. Chuyển bộ lọc sang Tất cả bình luận
    client.eval_js("""
    (() => {
        const filterBtn = Array.from(document.querySelectorAll('div[role="button"], span')).find(el => el.innerText && (el.innerText.trim().startsWith('Phù hợp nhất') || el.innerText.trim().startsWith('Bình luận hàng đầu')));
        if (filterBtn) {
            filterBtn.click();
            setTimeout(() => {
                const opt = Array.from(document.querySelectorAll('div[role="menuitem"], div[role="button"], span')).find(el => el.innerText && el.innerText.includes('Tất cả bình luận'));
                if (opt) opt.click();
            }, 300);
        }
    })()
    """)
    time.sleep(1.0)

    # 2. Mở rộng bình luận 5 đợt (cả câu trả lời con)
    for _ in range(5):
        client.eval_js("""
        (() => {
            window.scrollBy(0, 700);
            const buttons = Array.from(document.querySelectorAll('div[role="button"], span[role="button"], span'));
            for (const b of buttons) {
                const t = b.innerText ? b.innerText.trim() : '';
                if (t.includes('Xem thêm bình luận') || t.includes('Xem các bình luận trước') || t.includes('bình luận trước') || t.includes('câu trả lời')) {
                    try { b.click(); } catch(e) {}
                }
            }
            const seeMores = Array.from(document.querySelectorAll('div[dir="auto"] span, span[dir="auto"]'));
            for (const s of seeMores) {
                if (s.innerText && s.innerText.trim() === 'Xem thêm') {
                    try { s.click(); } catch(e) {}
                }
            }
        })()
        """)
        time.sleep(0.8)

    # 3. Trích xuất
    comments = client.eval_js(EXTRACT_JS) or []
    clean_comments = []
    for c in comments:
        c["source_url"] = f"https://www.facebook.com/groups/{group_slug}/posts/{post_id}/"
        c["is_truncated"] = False
        clean_comments.append(c)

    new_comments = [c for c in clean_comments if c["id"] not in existing_comment_ids]
    if clean_comments:
        snap_file = capture_dir / f"{prefix}_snapshot{snapshot_idx:03d}.json"
        snap_file.write_text(json.dumps(clean_comments, ensure_ascii=False, indent=2), encoding="utf-8")
        existing_comment_ids.update(c["id"] for c in clean_comments)
        return len(new_comments), snapshot_idx + 1

    return 0, snapshot_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["ptit_2k5", "ptit_group_584397217391365", "both"], default="both",
                        help="Nguồn cần thu thập")
    parser.add_argument("--target", type=int, default=2000,
                        help="Mục tiêu số bình luận (mặc định 2000)")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS,
                        help="Danh sách năm cần cào (ví dụ: 2024 2023 2025 2022)")
    parser.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS,
                        help="Danh sách từ khóa chủ đề sinh viên PTIT")
    args = parser.parse_args()

    sources = ["ptit_2k5", "ptit_group_584397217391365"] if args.source == "both" else [args.source]
    capture_dir = Path("data/raw/browser_capture")
    capture_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("   HỆ THỐNG CÀO DỮ LIỆU TỐC ĐỘ CAO THEO NĂM & CHỦ ĐỀ PTIT", flush=True)
    print(f"   Mục tiêu: {args.target} bình luận hợp lệ")
    print(f"   Khoảng thời gian: {args.years}")
    print(f"   Chủ đề: {args.topics[:6]} ... (+ {len(args.topics)-6} chủ đề khác)")
    print("=" * 70, flush=True)

    existing_posts = get_existing_post_ids()
    existing_comment_ids = get_existing_comment_ids()
    print(f"[*] Dữ liệu hiện có: {len(existing_comment_ids)} bình luận từ {len(existing_posts)} bài viết.", flush=True)

    client = CDPClient()
    client.connect()

    total_scraped_posts = 0

    try:
        for year in args.years:
            print(f"\n=======================================================", flush=True)
            print(f"   >>> BẮT ĐẦU QUÉT KHOẢNG THỜI GIAN: NĂM {year} <<<", flush=True)
            print(f"=======================================================", flush=True)

            for topic in args.topics:
                for src_key in sources:
                    grp = GROUPS[src_key]
                    slug = grp["slug"]
                    prefix = grp["prefix"]
                    snapshot_idx = get_next_snapshot_idx(capture_dir, prefix)

                    print(f"\n[Năm {year} | Chủ đề: '{topic}' | Nhóm: {grp['name']}]", flush=True)
                    new_post_ids = search_posts_by_year_and_topic(client, slug, year, topic, existing_posts, max_scrolls=3)
                    print(f"  -> Phát hiện {len(new_post_ids)} bài viết mới chưa từng cào.")

                    for p_idx, post_id in enumerate(new_post_ids):
                        existing_posts.add(post_id)
                        print(f"  [{p_idx+1}/{len(new_post_ids)}] Đang cào bài {post_id}...", end="", flush=True)
                        new_cnt, snapshot_idx = scrape_post_comments(client, slug, prefix, post_id, existing_comment_ids, capture_dir, snapshot_idx)
                        print(f" +{new_cnt} bình luận mới.", flush=True)
                        total_scraped_posts += 1

                        # Cứ mỗi 3 bài hoặc khi có bình luận mới, đồng bộ ngay
                        if new_cnt > 0 or total_scraped_posts % 3 == 0:
                            try:
                                report = sync("configs/collection.yaml", "data/raw/browser_capture", "data/raw/ptit_sources_v1", target=args.target)
                                cur = report["valid_unique_ids"]
                                pct = cur / args.target * 100
                                print(f"    📊 [Tiến độ: {cur}/{args.target} ({pct:.1f}%) | {report['posts']} bài | Còn thiếu {max(0, args.target - cur)}]", flush=True)

                                if report["target_reached"] or cur >= args.target:
                                    print(f"\n🎉🎉🎉 CHÚC MỪNG! ĐÃ ĐẠT VƯỢT MỤC TIÊU {args.target} BÌNH LUẬN! 🎉🎉🎉", flush=True)
                                    return
                            except Exception as sync_err:
                                print(f"    ⚠️ Đồng bộ tạm thời gặp lỗi: {sync_err}. Dữ liệu snapshot vẫn được bảo toàn.", flush=True)

                        time.sleep(1.0)

    finally:
        client.close()
        # Đồng bộ lần cuối
        try:
            sync("configs/collection.yaml", "data/raw/browser_capture", "data/raw/ptit_sources_v1", target=args.target)
        except Exception as sync_err:
            print(f"⚠️ Đồng bộ lần cuối gặp lỗi: {sync_err}", flush=True)


if __name__ == "__main__":
    main()
