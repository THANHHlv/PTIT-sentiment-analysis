"""Kiểm tra chi tiết element chứa text của comment."""
import json
import sys
import urllib.request
import websocket

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")


def main():
    tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
    tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=10, suppress_origin=True)

    info = ws.send(json.dumps({
        "id": 1,
        "method": "Runtime.evaluate",
        "params": {
            "expression": """
            (() => {
                const link = document.querySelector('a[href*="comment_id=1293221329553708"]');
                let cur = link;
                for (let i = 0; i < 7; i++) cur = cur.parentElement;
                const dirs = Array.from(cur.querySelectorAll('div[dir="auto"], span[dir="auto"]'))
                                  .map(el => ({tag: el.tagName, text: el.innerText}));
                return dirs;
            })()
            """,
            "returnByValue": True
        }
    }))
    res = json.loads(ws.recv())
    dirs = res.get("result", {}).get("result", {}).get("value", [])
    for d in dirs:
        print(f"<{d['tag']}>: {repr(d['text'])}")

    ws.close()


if __name__ == "__main__":
    main()
