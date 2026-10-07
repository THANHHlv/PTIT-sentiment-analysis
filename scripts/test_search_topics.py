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

import csv
with open("data/raw/ptit_sources_v1/comments.csv", encoding="utf-8") as f:
    existing_posts = set(r["post_id"] for r in csv.DictReader(f))

topics = ["review", "thầy", "kí túc xá", "học phí", "môn", "thi"]
print("Total existing posts in comments.csv:", len(existing_posts))

for topic in topics:
    search_url = f"https://www.facebook.com/groups/2k5ptit/search/?q={urllib.parse.quote(topic)}"
    print(f"\nSearching topic '{topic}': {search_url}")
    send_cmd("Page.navigate", {"url": search_url})
    time.sleep(4.0)
    
    # Cuộn 3 lần
    for _ in range(3):
        eval_js("window.scrollBy(0, 1500);")
        time.sleep(1.5)
        
    ids = eval_js("""
    (() => {
        const anchors = Array.from(document.querySelectorAll('a[href*="/posts/"], a[href*="/permalink/"], a[href*="multi_permalinks="], a[href*="story_fbid="]'));
        const s = new Set();
        for (const a of anchors) {
            const m1 = a.href.match(/(?:posts|permalink)\\/(\\d+)/);
            if (m1) s.add(m1[1]);
            const m2 = a.href.match(/[?&](?:story_fbid|fbid|multi_permalinks)=(\\d+)/);
            if (m2) s.add(m2[1]);
        }
        return Array.from(s);
    })()
    """) or []
    
    new_ids = [p for p in ids if p not in existing_posts]
    print(f"  Found {len(ids)} posts ({len(new_ids)} NEW posts)")
    print("  Sample new post IDs:", new_ids[:5])
    break

ws.close()
