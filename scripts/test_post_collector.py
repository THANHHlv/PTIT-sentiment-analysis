"""Thử nghiệm thu thập toàn bộ bình luận của 1 bài viết cụ thể."""
import json
import sys
import time
import urllib.request
import websocket

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")


def get_tab():
    tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
    return next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))


def eval_js(ws, expr):
    ws.send(json.dumps({
        "id": 1,
        "method": "Runtime.evaluate",
        "params": {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": True
        }
    }))
    res = json.loads(ws.recv())
    return res.get("result", {}).get("result", {}).get("value")


def navigate(ws, url):
    ws.send(json.dumps({
        "id": 2,
        "method": "Page.navigate",
        "params": {"url": url}
    }))
    json.loads(ws.recv())


EXTRACT_POST_COMMENTS_JS = """
(() => {
    // 1. Click các nút mở rộng bình luận (CHỈ click nút có chữ mở rộng bình luận)
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

    // 3. Trích xuất bình luận
    const comments = [];
    const seenIds = new Set();
    const links = Array.from(document.querySelectorAll('a[href*="comment_id="]'));

    // Lấy postId từ URL
    const urlMatch = window.location.href.match(/posts\\/(\\d+)/);
    const postId = urlMatch ? urlMatch[1] : '';

    for (const a of links) {
        const cidMatch = a.href.match(/comment_id=(\\d+)/);
        if (!cidMatch) continue;
        const commentId = cidMatch[1];
        if (seenIds.has(commentId)) continue;

        // Container
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
            if (t === 'Thích' || t === 'Trả lời' || t === 'Chia sẻ' || t.includes('phút') || t.includes('giờ') || t.includes('ngày')) continue;
            if (div.querySelector('a') && div.querySelector('a').innerText === t) continue;
            text = t;
            break;
        }

        // Bỏ '… Xem thêm' nếu còn
        text = text.replace(/[…\\.]+\\s*Xem thêm$/g, '').trim();

        if (text && text.length > 0 && commentId.match(/^\\d+$/)) {
            seenIds.add(commentId);
            comments.push({
                id: commentId,
                post_id: postId,
                source_url: `https://www.facebook.com/groups/2k5ptit/posts/${postId}/`,
                comment_url: a.href,
                text: text,
                is_truncated: false
            });
        }
    }

    return comments;
})()
"""


def collect_single_post(post_id):
    tab = get_tab()
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], suppress_origin=True)
    post_url = f"https://www.facebook.com/groups/2k5ptit/posts/{post_id}/"
    print(f"Đang mở bài viết: {post_url}")
    navigate(ws, post_url)
    time.sleep(3.5)

    # Thử mở rộng bình luận 3 lần
    for i in range(3):
        eval_js(ws, "window.scrollBy(0, 400);")
        time.sleep(1.0)
        comments = eval_js(ws, EXTRACT_POST_COMMENTS_JS)
        print(f"  Lần mở rộng {i + 1}: lấy được {len(comments) if comments else 0} bình luận")
        time.sleep(1.0)

    comments = eval_js(ws, EXTRACT_POST_COMMENTS_JS)
    ws.close()
    return comments


def main():
    post_id = "1293290179546823"
    comments = collect_single_post(post_id)
    print(f"\nTổng kết thu được {len(comments)} bình luận từ bài {post_id}:")
    for c in comments:
        print(f"  [id: {c['id']}] {repr(c['text'])}")


if __name__ == "__main__":
    main()
