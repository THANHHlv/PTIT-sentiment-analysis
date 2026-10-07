import urllib.request
import json
import websocket

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

info = eval_js("""
(() => {
    const allA = Array.from(document.querySelectorAll('a')).map(a => a.href);
    const postUrls = allA.filter(h => h.includes('posts') || h.includes('permalink') || h.includes('story_fbid') || h.includes('multi_permalinks'));
    return {
        currentUrl: window.location.href,
        totalA: allA.length,
        postUrls: postUrls
    };
})()
""")

print("Current URL:", info["currentUrl"])
print("Total anchors:", info["totalA"])
print("Post URLs count:", len(info["postUrls"]))
for u in info["postUrls"]:
    print(" ", u)

ws.close()
