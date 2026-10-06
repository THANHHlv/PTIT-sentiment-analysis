import urllib.request
import json
import websocket
import time
import sys

for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8")

tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=10, suppress_origin=True)

msg_id = 1
def send_cmd(method, params):
    global msg_id
    msg_id += 1
    ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
    res = json.loads(ws.recv())
    return res

def eval_js(js):
    return send_cmd("Runtime.evaluate", {"expression": js, "returnByValue": True, "awaitPromise": True}).get("result", {}).get("result", {}).get("value")

print("--- Testing CDP MouseWheel Scroll on Facebook Feed ---")
for i in range(10):
    # Send real mouse wheel event
    send_cmd("Input.dispatchMouseEvent", {
        "type": "mouseWheel",
        "x": 600,
        "y": 600,
        "deltaX": 0,
        "deltaY": 1200
    })
    time.sleep(1.5)
    
    st = eval_js("""
    (() => {
        const anchors = Array.from(document.querySelectorAll('a[href*="/posts/"], a[href*="/permalink/"], a[href*="multi_permalinks="], a[href*="story_fbid="]'));
        const ids = new Set();
        for (const a of anchors) {
            const m1 = a.href.match(/(?:posts|permalink)\\/(\\d+)/);
            if (m1) ids.add(m1[1]);
            const m2 = a.href.match(/[?&](?:story_fbid|fbid|multi_permalinks)=(\\d+)/);
            if (m2) ids.add(m2[1]);
        }
        return {
            posts: ids.size,
            scrollY: window.scrollY,
            docHeight: document.body.scrollHeight,
            feedPresent: Boolean(document.querySelector('[role="feed"]')),
            articles: document.querySelectorAll('div[role="article"]').length
        };
    })()
    """)
    print(f"Wheel {i+1}: posts={st['posts']}, articles={st['articles']}, scrollY={st['scrollY']}, docHeight={st['docHeight']}")

ws.close()
