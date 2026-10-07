"""Thu thập danh sách post_id từ feed của group."""
import json
import re
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
            "returnByValue": True
        }
    }))
    res = json.loads(ws.recv())
    return res.get("result", {}).get("result", {}).get("value")


def collect_post_ids(scroll_steps=10):
    tab = get_tab()
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], suppress_origin=True)

    found_posts = set()
    print(f"Đang cuộn feed {scroll_steps} lần để thu thập post_id...")

    for step in range(scroll_steps):
        eval_js(ws, "window.scrollBy(0, 1500);")
        time.sleep(1.5)

        # Trích xuất post_ids từ DOM hiện tại
        post_ids = eval_js(ws, """
        (() => {
            const ids = new Set();
            const anchors = Array.from(document.querySelectorAll('a[href*="/posts/"], a[href*="/permalink/"], a[href*="multi_permalinks="]'));
            for (const a of anchors) {
                const m1 = a.href.match(/groups\\/[^/]+\\/posts\\/(\\d+)/);
                if (m1) ids.add(m1[1]);
                const m2 = a.href.match(/permalink\\/(\\d+)/);
                if (m2) ids.add(m2[1]);
                const m3 = a.href.match(/multi_permalinks=(\\d+)/);
                if (m3) ids.add(m3[1]);
            }
            return Array.from(ids);
        })()
        """)
        if post_ids:
            before = len(found_posts)
            found_posts.update(post_ids)
            print(f"  Bước {step + 1}/{scroll_steps}: tìm thấy {len(post_ids)} posts trên DOM (tổng unique: {len(found_posts)})")

    ws.close()
    return sorted(found_posts)


def main():
    posts = collect_post_ids(12)
    print(f"\nTổng cộng tìm thấy {len(posts)} bài viết:")
    for p in posts[:20]:
        print(f"  https://www.facebook.com/groups/2k5ptit/posts/{p}/")


if __name__ == "__main__":
    main()
