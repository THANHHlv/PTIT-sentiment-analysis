import urllib.request
import json
import websocket
import sys

for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8")

tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
print("Active Tab URL:", tab.get("url"))
print("Active Tab Title:", tab.get("title"))

ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=10, suppress_origin=True)

def eval_js(js):
    ws.send(json.dumps({
        "id": 1,
        "method": "Runtime.evaluate",
        "params": {"expression": js, "returnByValue": True}
    }))
    res = json.loads(ws.recv())
    return res.get("result", {}).get("result", {}).get("value")

info = eval_js("""
(() => {
    const allLinks = Array.from(document.querySelectorAll('a')).map(a => a.href);
    const postLinks = allLinks.filter(h => h.includes('/posts/') || h.includes('/permalink/') || h.includes('multi_permalinks=') || h.includes('story_fbid='));
    const bodyLen = document.body.innerText.length;
    const bodySnippet = document.body.innerText.slice(0, 300);
    return {
        totalLinks: allLinks.length,
        postLinksCount: postLinks.length,
        samples: postLinks.slice(0, 15),
        bodyLen: bodyLen,
        bodySnippet: bodySnippet
    };
})()
""")

print("\n--- DOM Inspection ---")
print("Total links:", info["totalLinks"])
print("Post links count:", info["postLinksCount"])
print("Body length:", info["bodyLen"])
print("Body snippet:\n", info["bodySnippet"])
print("\nSample post links:")
for s in info["samples"]:
    print(" ", s)

ws.close()
