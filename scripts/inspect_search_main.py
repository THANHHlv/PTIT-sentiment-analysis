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
    const main = document.querySelector('div[role="main"]') || document.body;
    const allA = Array.from(main.querySelectorAll('a'));
    // Tìm các link chứa số ID bài viết
    const results = [];
    for (const a of allA) {
        const text = a.innerText.trim();
        const href = a.href;
        // Bỏ các link sidebar
        if (href.includes('/groups/2k5ptit') || href.includes('/groups/584397217391365')) {
            results.push({ text, href: href.slice(0, 150) });
        }
    }
    return results;
})()
"""

ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": expr, "returnByValue": True}}))
while True:
    r = json.loads(ws.recv())
    if r.get("id") == 1:
        res = r.get("result", {}).get("result", {}).get("value", [])
        print(f"Total matching group links in main: {len(res)}")
        for x in res[:15]:
            print(f"  [{x['text']}] -> {x['href']}")
        break

ws.close()
