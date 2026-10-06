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

def send_cmd(method, params):
    ws.send(json.dumps({"id": 1, "method": method, "params": params}))
    res = json.loads(ws.recv())
    return res

def eval_js(js):
    return send_cmd("Runtime.evaluate", {"expression": js, "returnByValue": True, "awaitPromise": True}).get("result", {}).get("result", {}).get("value")

feed_url = "https://www.facebook.com/groups/584397217391365?sorting_setting=CHRONOLOGICAL"
print("Navigating to:", feed_url)
send_cmd("Page.navigate", {"url": feed_url})
time.sleep(4.0)

for step in range(5):
    eval_js("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2.0)
    res = eval_js("""
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
            postIds: Array.from(ids),
            totalAnchors: anchors.length,
            scrollY: window.scrollY,
            docHeight: document.body.scrollHeight
        };
    })()
    """)
    print(f"Step {step+1}: postIds={len(res['postIds'])}, scrollY={res['scrollY']}, docHeight={res['docHeight']}")
    print("  Posts:", res["postIds"][:5])

ws.close()
