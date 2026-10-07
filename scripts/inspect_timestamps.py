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

timestamps = eval_js("""
(() => {
    // Tìm tất cả các span/a có chứa thông tin thời gian đăng bài
    const links = Array.from(document.querySelectorAll('a')).map(a => ({
        text: a.innerText.trim(),
        href: a.href,
        ariaLabel: a.getAttribute('aria-label')
    })).filter(x => x.text && (x.text.includes('giờ') || x.text.includes('ngày') || x.text.includes('Tháng') || x.text.includes('phút')));
    return links;
})()
""")

print("Found timestamp links:", len(timestamps))
for t in timestamps:
    print(" ", t)

ws.close()
