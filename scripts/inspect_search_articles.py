import urllib.request
import json
import websocket
import sys

for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8")

tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], suppress_origin=True)

expr = """
(() => {
    // Tìm các container bài viết trong kết quả tìm kiếm
    const allDivs = Array.from(document.querySelectorAll('div[role="feed"] > div, div[role="article"]'));
    return allDivs.map((d, idx) => {
        const text = d.innerText ? d.innerText.slice(0, 200).replace(/\n/g, ' | ') : '';
        const anchors = Array.from(d.querySelectorAll('a')).map(a => ({
            text: a.innerText.trim(),
            href: a.href,
            ariaLabel: a.getAttribute('aria-label') || ''
        }));
        return { idx, text, anchors };
    }).filter(x => x.text.length > 20).slice(0, 10);
})()
"""

ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": expr, "returnByValue": True}}))
while True:
    r = json.loads(ws.recv())
    if r.get("id") == 1:
        res = r.get("result", {}).get("result", {}).get("value", [])
        print(f"Total matching items: {len(res)}")
        for item in res:
            print(f"\n--- Item {item['idx']} ---")
            print("Text:", item["text"][:120])
            for a in item["anchors"]:
                print(f"  [A] text='{a['text']}' label='{a['ariaLabel']}' href='{a['href'][:120]}'")
        break

ws.close()
