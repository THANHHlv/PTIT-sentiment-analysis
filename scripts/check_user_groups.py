"""Tìm các nhóm PTIT mà người dùng đã tham gia."""
import json
import sys
import time
import urllib.request
import websocket

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")


def main():
    tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json/list").read())
    tab = next(t for t in tabs if t.get("type") == "page" and "facebook.com" in t.get("url", ""))
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=15, suppress_origin=True)

    print("Đang kiểm tra danh sách nhóm của người dùng...")
    ws.send(json.dumps({
        "id": 1,
        "method": "Page.navigate",
        "params": {"url": "https://www.facebook.com/groups/joins/"}
    }))
    time.sleep(3.5)

    info = ws.send(json.dumps({
        "id": 2,
        "method": "Runtime.evaluate",
        "params": {
            "expression": """
            (() => {
                const anchors = Array.from(document.querySelectorAll('a[href*="/groups/"]'));
                const groups = [];
                const seen = new Set();
                for (const a of anchors) {
                    const text = a.innerText.trim();
                    const m = a.href.match(/facebook\\.com\\/groups\\/([^/?]+)/);
                    if (m && text && !seen.has(m[1])) {
                        seen.add(m[1]);
                        groups.push({slug: m[1], title: text, url: a.href});
                    }
                }
                return groups;
            })()
            """,
            "returnByValue": True
        }
    }))
    res = json.loads(ws.recv())
    groups = res.get("result", {}).get("result", {}).get("value", [])

    print(f"Tổng số nhóm tìm thấy: {len(groups)}")
    ptit = [g for g in groups if "ptit" in g["title"].lower() or "bưu chính" in g["title"].lower() or "ptit" in g["slug"].lower()]
    print(f"Nhóm liên quan đến PTIT ({len(ptit)}):")
    for g in ptit:
        print(f"  - {g['title']} ({g['slug']}) -> {g['url']}")

    ws.close()


if __name__ == "__main__":
    main()
