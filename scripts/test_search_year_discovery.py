import urllib.request
import json
import websocket
import sys
import base64
import re
import time

for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8")

def make_year_filter(year):
    inner = json.dumps({
        "start_year": str(year),
        "start_month": f"{year}-1",
        "end_year": str(year),
        "end_month": f"{year}-12",
        "start_day": f"{year}-1-1",
        "end_day": f"{year}-12-31"
    })
    middle = json.dumps({"name": "creation_time", "args": inner})
    return base64.b64encode(json.dumps({"rp_creation_time:0": middle}).encode()).decode()

tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=20, suppress_origin=True)

msg_id = 0
def send_cmd(method, params=None):
    global msg_id
    msg_id += 1
    ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == msg_id:
            return r

def eval_js(expr):
    return send_cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}).get("result", {}).get("result", {}).get("value")

# Load existing posts from comments.csv
import csv
with open("data/raw/ptit_sources_v1/comments.csv", encoding="utf-8") as f:
    existing_posts = set(r["post_id"] for r in csv.DictReader(f))

print(f"Existing posts in dataset: {len(existing_posts)}")

year = 2024
kw = "thi"
filter_b64 = make_year_filter(year)
search_url = f"https://www.facebook.com/groups/450310163844833/search/?q={urllib.parse.quote(kw)}&filters={filter_b64}"
print(f"\nSearching Year {year} + Keyword '{kw}':\n{search_url}")

send_cmd("Page.navigate", {"url": search_url})
time.sleep(5.0)

# Scroll 3 times down to load more search results
for s in range(3):
    eval_js("window.scrollBy(0, 1500);")
    time.sleep(1.5)

html = eval_js("document.body.innerHTML")
matches = set(re.findall(r'(?:post_id|story_fbid|top_level_post_id)[\'"]?:[\'"]?(\d+)[\'"]?', html))
# Exclude the group ID itself
matches.discard("450310163844833")
matches.discard("584397217391365")

new_posts = [p for p in matches if p not in existing_posts]
print(f"Total post IDs found: {len(matches)}")
print(f"NEW post IDs (not yet in dataset): {len(new_posts)}")
print(f"Sample new post IDs: {new_posts[:8]}")

ws.close()
