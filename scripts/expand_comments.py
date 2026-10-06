"""Thử mở rộng các bình luận trên trang."""
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

    # Click các nút mở rộng
    click_js = """
    (() => {
        let clicked = 0;
        // 1. Click các nút "Xem thêm bình luận" hoặc "Xem câu trả lời"
        const candidates = Array.from(document.querySelectorAll('div[role="button"], span[role="button"], span'));
        for (const el of candidates) {
            const t = el.innerText ? el.innerText.trim() : '';
            if (t.includes('Xem thêm bình luận') || t.includes('Xem tất cả') || 
                t.includes('Xem thêm câu trả lời') || t.match(/Xem \\d+ câu trả lời/)) {
                try {
                    el.click();
                    clicked++;
                } catch(e) {}
            }
        }
        
        // 2. Click "Xem thêm" trong nội dung bình luận dài
        const seeMoreSpans = Array.from(document.querySelectorAll('div[dir="auto"] span, span[dir="auto"]'));
        for (const s of seeMoreSpans) {
            if (s.innerText && s.innerText.trim() === 'Xem thêm') {
                try {
                    s.click();
                    clicked++;
                } catch(e) {}
            }
        }
        
        return clicked;
    })()
    """

    ws.send(json.dumps({
        "id": 1,
        "method": "Runtime.evaluate",
        "params": {
            "expression": click_js,
            "returnByValue": True
        }
    }))
    res = json.loads(ws.recv())
    clicked = res.get("result", {}).get("result", {}).get("value", 0)
    print(f"Đã click mở rộng {clicked} nút.")

    ws.close()


if __name__ == "__main__":
    main()
