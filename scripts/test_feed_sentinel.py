import urllib.request
import json
import websocket
import time

tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], suppress_origin=True)

msg_id = 0
def send_cmd(method, params=None):
    global msg_id
    msg_id += 1
    ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
    while True:
        res = json.loads(ws.recv())
        if res.get("id") == msg_id:
            return res

def eval_js(js):
    return send_cmd("Runtime.evaluate", {"expression": js, "returnByValue": True, "awaitPromise": True}).get("result", {}).get("result", {}).get("value")

print("--- Testing scrollIntoView on Feed Bottom ---")
for step in range(8):
    eval_js("""
    (() => {
        const feed = document.querySelector('[role="feed"]');
        if (feed && feed.lastElementChild) {
            feed.lastElementChild.scrollIntoView(false);
        } else {
            window.scrollBy(0, 1500);
        }
    })()
    """)
    time.sleep(2.5)
    
    st = eval_js("""
    (() => {
        const feed = document.querySelector('[role="feed"]');
        const anchors = Array.from(document.querySelectorAll('a[href*="/posts/"], a[href*="/permalink/"]'));
        const ids = new Set();
        for (const a of anchors) {
            const m = a.href.match(/(?:posts|permalink)\\/(\\d+)/);
            if (m) ids.add(m[1]);
        }
        return {
            articles: feed ? feed.querySelectorAll('[role="article"]').length : 0,
            postIds: ids.size,
            docHeight: document.body.scrollHeight,
            scrollY: window.scrollY
        };
    })()
    """)
    print(f"Step {step+1}: articles={st['articles']}, uniquePostIds={st['postIds']}, docHeight={st['docHeight']}, scrollY={st['scrollY']}")

ws.close()
