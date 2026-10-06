"""Kiểm tra chi tiết DOM của tab Facebook."""
import json
import sys
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

    info = eval_js(ws, """
    (() => {
        const allA = Array.from(document.querySelectorAll('a')).map(a => a.href);
        const articles = document.querySelectorAll('div[role="article"]').length;
        const feed = document.querySelectorAll('[role="feed"]').length;
        const groupLinks = allA.filter(h => h.includes('groups') || h.includes('permalink') || h.includes('posts'));
        const bodySnippet = document.body.innerText.slice(0, 500);
        return {
            totalLinks: allA.length,
            articlesCount: articles,
            feedCount: feed,
            groupLinksCount: groupLinks.length,
            sampleGroupLinks: groupLinks.slice(0, 15),
            bodySnippet: bodySnippet
        };
    })()
    """)

    print(f"Total links: {info['totalLinks']}")
    print(f"Articles count: {info['articlesCount']}")
    print(f"Feed count: {info['feedCount']}")
    print(f"Group/post links: {info['groupLinksCount']}")
    print("Sample links:")
    for l in info["sampleGroupLinks"]:
        print(" ", l)
    print("\nBody snippet:")
    print(info["bodySnippet"][:200])

    ws.close()


if __name__ == "__main__":
    main()
