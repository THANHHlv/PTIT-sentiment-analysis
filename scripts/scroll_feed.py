"""Thử cuộn trang và tìm bài viết trong feed."""
import json
import sys
import time
import urllib.request
import websocket

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")


def get_facebook_tab():
    resp = urllib.request.urlopen("http://127.0.0.1:9222/json/list", timeout=3)
    tabs = json.loads(resp.read().decode())
    for t in tabs:
        if t.get("type") == "page" and "facebook.com" in t.get("url", ""):
            return t
    return None


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
    if "result" in res and "result" in res["result"]:
        return res["result"]["result"].get("value")
    return None


def main():
    tab = get_facebook_tab()
    if not tab:
        print("Không tìm thấy tab Facebook.")
        return

    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=10, suppress_origin=True)

    print("Đang cuộn trang xuống để tải feed bài viết...")
    for step in range(5):
        eval_js(ws, "window.scrollBy(0, 1200);")
        time.sleep(1.5)

    info = eval_js(ws, """
    (() => {
        const articles = document.querySelectorAll('div[role="feed"] div[role="article"]');
        const postData = [];
        for (const art of articles) {
            // Tìm tất cả a bên trong article
            const anchors = Array.from(art.querySelectorAll('a'));
            let postUrl = '';
            let postId = '';
            for (const a of anchors) {
                // Link bài viết thường có dạng /posts/ID hoặc /permalink/ID hoặc multi_permalinks
                const m = a.href.match(/groups\\/(?:2k5ptit|450310163844833)\\/posts\\/(\\d+)/);
                if (m) {
                    postId = m[1];
                    postUrl = m[0];
                    break;
                }
            }
            if (!postId) {
                for (const a of anchors) {
                    const m = a.href.match(/permalink\\/(\\d+)/) || a.href.match(/story_fbid=(\\d+)/);
                    if (m) {
                        postId = m[1];
                        postUrl = a.href;
                        break;
                    }
                }
            }
            // Tìm nút bình luận hoặc số bình luận
            const commentButtons = Array.from(art.querySelectorAll('[aria-label*="bình luận"], [aria-label*="Bình luận"], [aria-label*="comment"], [aria-label*="Comment"]'));
            // Lấy một đoạn text bài viết
            const textEl = art.querySelector('div[dir="auto"]');
            const snippet = textEl ? textEl.innerText.slice(0, 80) : '';
            
            postData.push({
                postId: postId,
                postUrl: postUrl,
                commentButtons: commentButtons.length,
                snippet: snippet
            });
        }
        return {
            totalArticles: articles.length,
            posts: postData
        };
    })()
    """)

    print(f"Tổng số articles tìm thấy: {info['totalArticles']}")
    for p in info["posts"]:
        print(f"  - Post ID: {p['postId'] or 'chưa tìm thấy'} | Comments: {p['commentButtons']} | Snippet: {p['snippet']}")

    ws.close()


if __name__ == "__main__":
    main()
