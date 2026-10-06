import urllib.request
import json
import websocket
import time

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

print("--- Testing EXTRACT_JS on current page ---")
comments = eval_js(EXTRACT_JS) or []
print(f"Extracted comments count: {len(comments)}")
for c in comments[:5]:
    print(" ", c)

debug_info = eval_js("""
(() => {
    const allLinks = Array.from(document.querySelectorAll('a[href*="comment_id="]'));
    const results = allLinks.map(a => {
        const replyMatch = a.href.match(/reply_comment_id=(\\d+)/);
        const cidMatch = a.href.match(/comment_id=(\\d+)/);
        const commentId = replyMatch ? replyMatch[1] : (cidMatch ? cidMatch[1] : null);
        return {
            id: commentId,
            isReply: Boolean(replyMatch),
            parentCid: cidMatch ? cidMatch[1] : null
        };
    });
    return results;
})()
""")

print("\nAll comment links with reply support:")
for r in debug_info:
    print(" ", r)

print("\nDebug Info:")
print("Total links:", debug_info["totalLinks"])
print("Comment links count:", debug_info["commentLinksCount"])
print("Expand buttons visible:", debug_info["expandButtons"])
print("Sample comment hrefs:")
for h in debug_info["sampleCommentHrefs"]:
    print(" ", h)

ws.close()
