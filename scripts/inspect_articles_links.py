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
    const articles = Array.from(document.querySelectorAll('div[role="article"]'));
    return articles.map((art, idx) => {
        const text = art.innerText.slice(0, 80).replace(/\\n/g, ' ');
        const links = Array.from(art.querySelectorAll('a')).map(a => ({
            text: a.innerText.trim(),
            href: a.href
        }));
        return { idx, text, links };
    });
})()
"""

ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": expr, "returnByValue": True}}))
while True:
    r = json.loads(ws.recv())
    if r.get("id") == 1:
        res = r.get("result", {}).get("result", {}).get("value", [])
        print(f"Total articles on page: {len(res)}")
        for a in res:
            print(f"\n[{a['idx']}] {a['text']}")
            for l in a["links"][:4]:
                print(f"   [{l['text']}] -> {l['href'][:110]}")
        break

ws.close()
