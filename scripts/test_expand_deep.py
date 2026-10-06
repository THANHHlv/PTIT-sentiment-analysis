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

import sys
sys.path.insert(0, ".")
from scripts.auto_collect_ptit import EXTRACT_JS

# Mở một bài viết có nhiều bình luận để test
test_post_url = "https://www.facebook.com/groups/2k5ptit/posts/1293290179546823/"
print("Opening post:", test_post_url)
send_cmd("Page.navigate", {"url": test_post_url})
time.sleep(4.0)

# Chuyển bộ lọc sang Tất cả bình luận
eval_js("""
(() => {
    const filterBtn = Array.from(document.querySelectorAll('div[role="button"], span')).find(el => el.innerText && (el.innerText.trim().startsWith('Phù hợp nhất') || el.innerText.trim().startsWith('Bình luận hàng đầu')));
    if (filterBtn) filterBtn.click();
})()
""")
time.sleep(1.0)
eval_js("""
(() => {
    const opt = Array.from(document.querySelectorAll('div[role="menuitem"], div[role="button"], span')).find(el => el.innerText && el.innerText.includes('Tất cả bình luận'));
    if (opt) opt.click();
})()
""")
time.sleep(1.5)

# Bấm mở rộng 5 lượt
for r in range(5):
    cnt = eval_js("""
    (() => {
        window.scrollBy(0, 800);
        let clicked = 0;
        const buttons = Array.from(document.querySelectorAll('div[role="button"], span[role="button"], span'));
        for (const b of buttons) {
            const t = b.innerText ? b.innerText.trim() : '';
            if (t.includes('Xem thêm bình luận') || t.includes('Xem các bình luận trước') || t.includes('bình luận trước') || t.includes('câu trả lời')) {
                try { b.click(); clicked++; } catch(e) {}
            }
        }
        return clicked;
    })()
    """)
    print(f"Round {r+1}: clicked {cnt} expand buttons")
    time.sleep(1.5)

comments = eval_js(EXTRACT_JS) or []
print(f"\nExtracted comments from test post: {len(comments)}")
for c in comments[:5]:
    print(" ", c.get("id"), ":", c.get("text")[:60])

ws.close()
