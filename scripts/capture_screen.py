import urllib.request
import json
import websocket
import base64

tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=10, suppress_origin=True)

ws.send(json.dumps({"id": 1, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
while True:
    res = json.loads(ws.recv())
    if res.get("id") == 1:
        break

data = res.get("result", {}).get("data")
if data:
    with open("data/chrome_screen.png", "wb") as f:
        f.write(base64.b64decode(data))
    print("Screenshot saved to data/chrome_screen.png")
else:
    print("Failed to capture screenshot")

ws.close()
