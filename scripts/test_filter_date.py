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
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=20, suppress_origin=True)

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

url = "https://www.facebook.com/groups/2k5ptit/search/?q=review"
print("Navigating to search page:", url)
send_cmd("Page.navigate", {"url": url})
time.sleep(5.0)

# Tìm nút "Ngày đăng" và click thử
date_btn = eval_js("""
(() => {
    const allSpans = Array.from(document.querySelectorAll('span, div[role="button"]'));
    const btn = allSpans.find(el => el.innerText && el.innerText.trim() === 'Ngày đăng');
    if (btn) {
        btn.click();
        return "Clicked Ngày đăng";
    }
    return "Not found Ngày đăng";
})()
""")
print("Date button click result:", date_btn)
time.sleep(1.5)

# Lấy các tùy chọn xuất hiện
options = eval_js("""
(() => {
    const all = Array.from(document.querySelectorAll('div[role="menuitem"], div[role="button"], span, div[role="radio"]')).map(el => el.innerText.trim()).filter(Boolean);
    return Array.from(new Set(all)).filter(t => t.includes('202') || t.includes('Năm') || t.includes('tháng') || t.includes('khoảng') || t.includes('Gần đây') || t.includes('Hôm nay'));
})()
""")
print("Date filter options:", options)

ws.close()
