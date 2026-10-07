"""Kiểm tra chi tiết cấu trúc bình luận trong article."""
import json
import sys
import urllib.request
import websocket

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")


def get_facebook_tab():
    resp = urllib.request.urlopen("http://127.0.0.1:9222/json/list", timeout=3)
    tabs = json.loads(resp.read().decode())
    for t in tabs:
        if t.get("type") == "page" and "facebook.com" in t.get("url", ""):
            return t
    return None


def eval_js(ws, expr):
    ws.send(json.dumps({
        "id": 1,
        "method": "Runtime.evaluate",
        "params": {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": True
        }
    }))
    res = json.loads(ws.recv())
    if "result" in res and "result" in res["result"]:
        return res["result"]["result"].get("value")
    return None


def main():
    tab = get_facebook_tab()
    if not tab:
        return

    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=10, suppress_origin=True)

    info = eval_js(ws, """
    (() => {
        // Tìm các comment elements
        const commentLinks = Array.from(document.querySelectorAll('a[href*="comment_id="]'));
        const samples = [];
        for (const a of commentLinks.slice(0, 5)) {
            // container chứa comment này
            let parent = a;
            for (let i = 0; i < 6; i++) {
                if (parent.parentElement) parent = parent.parentElement;
            }
            const textNodes = Array.from(parent.querySelectorAll('div[dir="auto"], span[dir="auto"]'))
                                  .map(el => el.innerText.trim())
                                  .filter(t => t.length > 0 && !['Thích', 'Trả lời', 'Chia sẻ'].includes(t));
            
            samples.push({
                href: a.href,
                textNodes: textNodes.slice(0, 3)
            });
        }
        
        // Tìm các nút "Xem thêm bình luận"
        const moreButtons = Array.from(document.querySelectorAll('div[role="button"], span'))
                                .map(b => b.innerText.trim())
                                .filter(t => t.includes('Xem thêm') || t.includes('bình luận khác') || t.includes('câu trả lời'));
                                
        return {
            totalCommentLinks: commentLinks.length,
            samples: samples,
            moreButtons: moreButtons.slice(0, 10)
        };
    })()
    """)

    print(f"Comment links count: {info['totalCommentLinks']}")
    for s in info["samples"]:
        print(f"Link: {s['href']}")
        print(f"Texts: {s['textNodes']}")
    print(f"\nMore buttons count: {len(info['moreButtons'])}")
    for b in info["moreButtons"]:
        print(f"  - {b}")

    ws.close()


if __name__ == "__main__":
    main()
