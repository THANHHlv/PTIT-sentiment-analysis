import urllib.request
import json
import websocket
import time
import urllib.parse
import sys

for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8")

tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=20, suppress_origin=True)

fb_b64 = "eyJycF9jcmVhdGlvbl90aW1lOjAiOiJ7XCJuYW1lXCI6XCJjcmVhdGlvbl90aW1lXCIsXCJhcmdzXCI6XCJ7XFxcInN0YXJ0X3llYXJcXFwiOlxcXCIyMDI0XFxcIixcXFwic3RhcnRfbW9udGhcXFwiOlxcXCIyMDI0LTFcXFwiLFxcXCJlbmRfeWVhclxcXCI6XFxcIjIwMjRcXFwiLFxcXCJlbmRfbW9udGhcXFwiOlxcXCIyMDI0LTEyXFxcIixcXFwic3RhcnRfZGF5XFxcIjpcXFwiMjAyNC0xLTFcXFwiLFxcXCJlbmRfZGF5XFxcIjpcXFwiMjAyNC0xMi0zMVxcXCJ9XCJ9In0%3D"
query = urllib.parse.quote("học")
target_url = f"https://www.facebook.com/groups/2k5ptit/search/?q={query}&filters={fb_b64}"
print("Navigating to:", target_url)

ws.send(json.dumps({"id": 1, "method": "Page.navigate", "params": {"url": target_url}}))
while True:
    r = json.loads(ws.recv())
    if r.get("id") == 1:
        break
time.sleep(5.0)

ws.send(json.dumps({"id": 2, "method": "Runtime.evaluate", "params": {
    "expression": "document.body.innerText.slice(0, 400)",
    "returnByValue": True
}}))
while True:
    r = json.loads(ws.recv())
    if r.get("id") == 2:
        print("Page text snippet:\n", r.get("result", {}).get("result", {}).get("value"))
        break

ws.close()
