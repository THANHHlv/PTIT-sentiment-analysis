"""Kiểm tra chi tiết DOM xung quanh một comment link."""
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
        return

    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=10, suppress_origin=True)

    info = eval_js(ws, """
    (() => {
        const link = document.querySelector('a[href*="comment_id=1293221329553708"]') || document.querySelector('a[href*="comment_id="]');
        if (!link) return {error: "no link found"};
        
        // Leo lên các cấp parent
        let cur = link;
        const parents = [];
        for (let i = 0; i < 8; i++) {
            if (!cur.parentElement) break;
            cur = cur.parentElement;
            parents.push({
                level: i + 1,
                tag: cur.tagName,
                role: cur.getAttribute('role'),
                className: cur.className.slice(0, 30),
                innerText: cur.innerText.slice(0, 150)
            });
        }
        return {
            linkHref: link.href,
            parents: parents
        };
    })()
    """)

    print("Link:", info.get("linkHref"))
    for p in info.get("parents", []):
        print(f"Level {p['level']} <{p['tag']} role='{p['role']}'>: {repr(p['innerText'])}")

    ws.close()


if __name__ == "__main__":
    main()
