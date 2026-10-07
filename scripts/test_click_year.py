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

print("Clicking 2024...")
res = eval_js("""
(() => {
    const opts = Array.from(document.querySelectorAll('div[role="radio"], div[role="button"], span, div[role="menuitemradio"]'));
    const opt = opts.find(el => el.innerText && el.innerText.trim() === '2024');
    if (opt) {
        opt.click();
        return "Clicked 2024";
    }
    return "2024 not found";
})()
""")
print("Click result:", res)
time.sleep(3.5)

print("New URL after clicking 2024:")
new_url = eval_js("window.location.href")
print("  URL:", new_url)

articles_text = eval_js("""
(() => {
    const main = document.querySelector('div[role="main"]') || document.body;
    const cards = Array.from(main.querySelectorAll('div[role="article"], div[role="feed"] > div'));
    return cards.map(c => c.innerText.slice(0, 150).replace(/\\n/g, ' ')).filter(t => t.length > 30).slice(0, 6);
})()
""")
print(f"\nArticles found in 2024 ({len(articles_text)}):")
for t in articles_text:
    print(" -", t)

ws.close()
