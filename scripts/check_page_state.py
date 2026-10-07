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
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], suppress_origin=True)
    ws.send(json.dumps({
        "id": 1,
        "method": "Runtime.evaluate",
        "params": {
            "expression": """
            (() => {
                const commentLinks = document.querySelectorAll('a[href*="comment_id="]').length;
                const articles = document.querySelectorAll('div[role="article"]').length;
                return {
                    url: window.location.href,
                    title: document.title,
                    commentLinks: commentLinks,
                    articles: articles
                };
            })()
            """,
            "returnByValue": True
        }
    }))
    res = json.loads(ws.recv())
    print(res.get("result", {}).get("result", {}).get("value"))
    ws.close()


if __name__ == "__main__":
    main()
