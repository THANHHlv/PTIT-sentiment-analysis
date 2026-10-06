"""Kiểm tra cuộn sâu feed với trigger scrollHeight."""
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
            "returnByValue": True
        }
    }))
    res = json.loads(ws.recv())
    return res.get("result", {}).get("result", {}).get("value")


def main():
    tab = get_tab()
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=20, suppress_origin=True)

    url = "https://www.facebook.com/groups/2k5ptit"
    print(f"Mở feed: {url}")
    ws.send(json.dumps({"id": 2, "method": "Page.navigate", "params": {"url": url}}))
    time.sleep(3.5)

    all_posts = set()
    for i in range(15):
        # Cuộn tới đáy
        eval_js(ws, "window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2.5)

        posts = eval_js(ws, """
        (() => {
            const anchors = Array.from(document.querySelectorAll('a[href*="/posts/"], a[href*="/permalink/"]'));
            const ids = [];
            for (const a of anchors) {
                const m = a.href.match(/(?:posts|permalink)\\/(\\d+)/);
                if (m) ids.push(m[1]);
            }
            return Array.from(new Set(ids));
        })()
        """)
        before = len(all_posts)
        if posts:
            all_posts.update(posts)
        print(f"Lần {i+1}: DOM có {len(posts) if posts else 0} posts | Tổng unique: {len(all_posts)} (+{len(all_posts) - before})")

    ws.close()


if __name__ == "__main__":
    main()
