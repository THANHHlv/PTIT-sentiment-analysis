import urllib.request
import json
import websocket
import sys

for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8")

tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
tab = next(t for t in tabs if t.get("type") == "page" and "2k5ptit" in t.get("url", ""))
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], suppress_origin=True)

# First go back to search
ws.send(json.dumps({"id": 1, "method": "Page.navigate", "params": {"url": "https://www.facebook.com/groups/2k5ptit/search/?q=review"}}))
while True:
    r = json.loads(ws.recv())
    if r.get("id") == 1:
        break

import time
time.sleep(4.0)

# Check all links and elements on search page
expr = """
(() => {
    // Look at all anchors anywhere on page
    const allA = Array.from(document.querySelectorAll('a'));
    const relevantA = allA.filter(a => {
        const h = a.href || '';
        return h.includes('posts') || h.includes('permalink') || h.includes('story_fbid') || h.includes('multi_permalinks') || h.includes('comment_id');
    }).map(a => ({ text: a.innerText.slice(0, 50), href: a.href }));
    
    // Also look for any post IDs in the entire page HTML using regex
    const html = document.body.innerHTML;
    const postMatches = Array.from(new Set(Array.from(html.matchAll(/(?:post_id|story_fbid|top_level_post_id)[\"']?:[\"']?(\\d+)[\"']?/g)).map(m => m[1])));
    const permalinkMatches = Array.from(new Set(Array.from(html.matchAll(/\\/posts\\/(\\d+)/g)).map(m => m[1])));

    return {
        relevantAnchors: relevantA,
        postMatches: postMatches.slice(0, 20),
        permalinkMatches: permalinkMatches.slice(0, 20)
    };
})()
"""

ws.send(json.dumps({"id": 2, "method": "Runtime.evaluate", "params": {"expression": expr, "returnByValue": True}}))
while True:
    r = json.loads(ws.recv())
    if r.get("id") == 2:
        print("Results:\n", json.dumps(r.get("result", {}).get("result", {}).get("value"), ensure_ascii=False, indent=2))
        break

ws.close()
