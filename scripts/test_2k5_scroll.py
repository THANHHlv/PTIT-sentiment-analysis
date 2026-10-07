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

url = "https://www.facebook.com/groups/2k5ptit"
print("Navigating to:", url)
send_cmd("Page.navigate", {"url": url})
time.sleep(5.0)

for step in range(5):
    eval_js("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2.5)
    info = eval_js("""
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
            postIdsCount: ids.size,
            sampleIds: Array.from(ids).slice(0, 10),
            docHeight: document.body.scrollHeight,
            scrollY: window.scrollY
        };
    })()
    """)
    print(f"Step {step+1}: postIdsCount={info['postIdsCount']}, docHeight={info['docHeight']}, scrollY={info['scrollY']}")

ws.close()
