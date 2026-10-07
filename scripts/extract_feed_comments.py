"""Trích xuất tất cả bình luận hiện có trong DOM Facebook tab qua CDP."""
import json
import re
import sys
import urllib.request
import websocket

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")


EXTRACT_JS = """
(() => {
    const comments = [];
    const seenIds = new Set();
    const links = Array.from(document.querySelectorAll('a[href*="comment_id="]'));

    for (const a of links) {
        // Lấy post_id và comment_id
        const postMatch = a.href.match(/groups\\/([^/]+)\\/posts\\/(\\d+)/);
        const cidMatch = a.href.match(/comment_id=(\\d+)/);
        if (!postMatch || !cidMatch) continue;

        const groupSlug = postMatch[1];
        const postId = postMatch[2];
        const commentId = cidMatch[1];

        if (seenIds.has(commentId)) continue;

        // Leo lên container
        let cur = a;
        let container = null;
        for (let i = 0; i < 8; i++) {
            if (!cur.parentElement) break;
            cur = cur.parentElement;
            // Container thường có text lớn hơn chỉ timestamp
            if (cur.innerText && cur.innerText.split('\\n').length >= 3) {
                container = cur;
            }
        }
        if (!container) continue;

        // Tìm div[dir="auto"] chứa text
        const textDivs = Array.from(container.querySelectorAll('div[dir="auto"]'));
        let commentText = '';
        for (const div of textDivs) {
            const t = div.innerText.trim();
            // Bỏ qua nếu là tên tác giả hoặc thời gian hoặc nút tương tác
            if (!t) continue;
            if (t === 'Thích' || t === 'Trả lời' || t === 'Chia sẻ' || t.includes('phút') || t.includes('giờ') || t.includes('ngày')) continue;
            // Kiểm tra xem div này có chứa link tác giả không
            if (div.querySelector('a') && div.querySelector('a').innerText === t) continue;
            commentText = t;
            break;
        }

        if (commentText && commentText.length > 0) {
            seenIds.add(commentId);
            comments.push({
                id: commentId,
                post_id: postId,
                source_url: `https://www.facebook.com/groups/${groupSlug}/posts/${postId}/`,
                comment_url: a.href,
                text: commentText,
                is_truncated: false
            });
        }
    }

    return comments;
})()
"""


def main():
    tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
    tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=15, suppress_origin=True)

    ws.send(json.dumps({
        "id": 1,
        "method": "Runtime.evaluate",
        "params": {
            "expression": EXTRACT_JS,
            "returnByValue": True
        }
    }))
    res = json.loads(ws.recv())
    comments = res.get("result", {}).get("result", {}).get("value", [])

    print(f"Tổng số bình luận trích xuất được từ DOM: {len(comments)}")
    posts = set(c["post_id"] for c in comments)
    print(f"Số bài viết liên quan: {len(posts)}")
    for c in comments[:8]:
        print(f"  [id: {c['id']}] (post: {c['post_id']}): {repr(c['text'])}")

    ws.close()


if __name__ == "__main__":
    main()
