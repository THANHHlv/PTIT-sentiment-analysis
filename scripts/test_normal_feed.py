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

url = "https://www.facebook.com/groups/584397217391365"
print("Navigating to normal group feed:", url)
send_cmd("Page.navigate", {"url": url})
time.sleep(5.0)

for i in range(5):
    eval_js("window.scrollBy(0, 1000);")
    time.sleep(1.5)
    st = eval_js("""
    (() => {
        const anchors = Array.from(document.querySelectorAll('a[href*="/posts/"], a[href*="/permalink/"]'));
        const ids = new Set();
        for (const a of anchors) {
            const m = a.href.match(/(?:posts|permalink)\\/(\\d+)/);
            if (m) ids.add(m[1]);
        }
        return {
            articles: document.querySelectorAll('div[role="article"]').length,
            posts: ids.size,
            sample: Array.from(ids).slice(0, 5)
        };
    })()
    """)
    print(f"Scroll {i+1}: articles={st['articles']}, posts={st['posts']}, samples={st['sample']}")

ws.close()
