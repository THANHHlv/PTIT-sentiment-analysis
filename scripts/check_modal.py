import urllib.request
import json
import websocket

tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], suppress_origin=True)

def eval_js(js):
    ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": js, "returnByValue": True}}))
    while True:
        res = json.loads(ws.recv())
        if res.get("id") == 1:
            return res.get("result", {}).get("result", {}).get("value")

res = eval_js("""
(() => {
    const dialogs = Array.from(document.querySelectorAll('[role="dialog"]')).map(d => ({
        ariaLabel: d.getAttribute('aria-label'),
        classes: d.className,
        rect: d.getBoundingClientRect()
    }));
    const feed = document.querySelector('[role="feed"]');
    const articles = document.querySelectorAll('[role="article"]');
    return {
        dialogs: dialogs,
        feedExists: Boolean(feed),
        feedArticles: feed ? feed.querySelectorAll('[role="article"]').length : 0,
        totalArticles: articles.length,
        links: Array.from(document.querySelectorAll('a[href*="/posts/"], a[href*="/permalink/"]')).map(a => a.href)
    };
})()
""")

print("Dialogs:", len(res["dialogs"]))
for d in res["dialogs"]:
    print("  Dialog:", d["ariaLabel"], d["rect"])
print("Feed exists:", res["feedExists"])
print("Feed articles:", res["feedArticles"])
print("Total articles:", res["totalArticles"])
print("Post links:", len(res["links"]))
for l in res["links"][:5]:
    print(" ", l)

ws.close()
