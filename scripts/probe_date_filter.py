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

filters = eval_js("""
(() => {
    const items = Array.from(document.querySelectorAll('div[role="button"], span, div')).filter(el => {
        const t = (el.innerText || '').trim();
        return t === 'Ngày đăng' || t === 'Năm' || t === 'Mới đây nhất' || t.includes('2024') || t.includes('2023') || t.includes('2025');
    }).map(el => ({ text: el.innerText.trim(), tag: el.tagName, role: el.getAttribute('role') }));
    return items.slice(0, 15);
})()
""")

print("Filter elements found:", filters)
ws.close()
