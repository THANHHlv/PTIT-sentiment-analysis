"""Hướng dẫn và script hỗ trợ thu thập thêm bình luận PTIT.

Script này tạo bookmarklet JavaScript mà người dùng dán vào Console
của trình duyệt khi đang xem một bài viết trong nhóm PTIT.
Bookmarklet sẽ thu thập tất cả bình luận hiển thị trên trang và
xuất ra file JSON theo đúng schema cần thiết.

Cách dùng:
    1. Đăng nhập Facebook trên Chrome
    2. Mở một bài viết trong nhóm PTIT
    3. Mở rộng tất cả bình luận (nhấn "Xem thêm bình luận...")
    4. Mở Console (F12 > Console)
    5. Dán nội dung bookmarklet và nhấn Enter
    6. Lưu file JSON xuất ra vào data/raw/browser_capture/ hoặc data/raw/incoming/
    7. Chạy import:
       $python scripts/import_new_comments.py --file <file_path> --source-id ptit_2k5
"""
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8')


BOOKMARKLET_JS = r"""
// === Thu thập bình luận PTIT từ trang Facebook hiện tại ===
// Dán vào Console (F12) khi đang xem một bài viết trong nhóm PTIT
// Mở rộng tất cả bình luận trước khi chạy

(function() {
    // Lấy post_id từ URL
    const urlMatch = window.location.href.match(/\/posts\/(\d+)/);
    if (!urlMatch) {
        alert('Không tìm thấy post_id trong URL. Hãy mở một bài viết cụ thể.');
        return;
    }
    const postId = urlMatch[1];

    // Lấy group slug từ URL để xây source_url
    const groupMatch = window.location.href.match(/\/groups\/([^/]+)/);
    const groupSlug = groupMatch ? groupMatch[1] : 'unknown';
    const sourceUrl = `https://www.facebook.com/groups/${groupSlug}/posts/${postId}/`;

    // Thu thập bình luận từ DOM
    // Facebook renders comments in various structures; try multiple selectors
    const commentElements = document.querySelectorAll('[aria-label*="Comment"], [aria-label*="Bình luận"], [data-testid="comment"]');

    // Fallback: tìm tất cả div có role="article" bên trong phần bình luận
    let containers = commentElements.length > 0 ? commentElements :
        document.querySelectorAll('div[role="article"]');

    const comments = [];
    const seenTexts = new Set();

    containers.forEach(container => {
        // Lấy text từ container
        const textElements = container.querySelectorAll('div[dir="auto"], span[dir="auto"]');
        let text = '';
        textElements.forEach(el => {
            const t = el.textContent.trim();
            if (t && t.length > 0 && !t.startsWith('Thích') && !t.startsWith('Trả lời')
                && !t.startsWith('Like') && !t.startsWith('Reply')) {
                text += (text ? '\n' : '') + t;
            }
        });

        if (!text || text.length === 0) return;

        // Lấy comment_id từ link permalink nếu có
        const permalinkEl = container.querySelector('a[href*="comment_id="]');
        let commentId = '';
        if (permalinkEl) {
            const cidMatch = permalinkEl.href.match(/comment_id=(\d+)/);
            commentId = cidMatch ? cidMatch[1] : '';
        }

        // Nếu không có comment_id, tạo hash từ text
        if (!commentId) {
            commentId = 'dom_' + postId + '_' + Array.from(text).reduce((h, c) =>
                Math.imul(31, h) + c.charCodeAt(0) | 0, 0).toString(36);
        }

        // Dedup
        const key = text.toLowerCase().replace(/\s+/g, ' ').trim();
        if (seenTexts.has(key)) return;
        seenTexts.add(key);

        comments.push({
            id: commentId,
            post_id: postId,
            source_url: sourceUrl,
            text: text,
            is_truncated: false
        });
    });

    if (comments.length === 0) {
        alert('Không tìm thấy bình luận nào. Hãy mở rộng tất cả bình luận trước.');
        return;
    }

    // Xuất file JSON
    const blob = new Blob([JSON.stringify(comments, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    a.href = url;
    a.download = `ptit_${groupSlug}_${postId}_${timestamp}.json`;
    a.click();
    URL.revokeObjectURL(url);

    alert(`Đã thu ${comments.length} bình luận từ bài ${postId}.\nFile đã tải xuống.`);
})();
"""


def generate_bookmarklet():
    """Tạo file bookmarklet để người dùng dùng trong Console."""
    output_dir = Path("tools/bookmarklets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lưu phiên bản đọc được
    output_path = output_dir / "collect_comments.js"
    output_path.write_text(BOOKMARKLET_JS.strip(), encoding="utf-8")
    print(f"Đã tạo bookmarklet tại: {output_path}")
    print()
    print("=" * 60)
    print("HƯỚNG DẪN THU THẬP BÌNH LUẬN PTIT")
    print("=" * 60)
    print()
    print("1. Đăng nhập Facebook trên Chrome")
    print("2. Mở một bài viết trong nhóm PTIT:")
    print("   - https://www.facebook.com/groups/2k5ptit")
    print("   - https://www.facebook.com/groups/584397217391365")
    print("3. Nhấn 'Xem thêm bình luận' nhiều lần để mở hết bình luận")
    print("4. Mở Console: nhấn F12 → chọn tab Console")
    print("5. Copy nội dung file bookmarklet và dán vào Console")
    print("6. Nhấn Enter → file JSON sẽ tự tải xuống")
    print("7. Di chuyển file JSON vào thư mục dự án:")
    print("   data/raw/browser_capture/ hoặc data/raw/incoming/")
    print("8. Nhập dữ liệu:")
    print('   .\\.venv\\Scripts\\python.exe scripts/import_new_comments.py \\')
    print("     --file data/raw/incoming/<tên_file>.json \\")
    print("     --source-id ptit_2k5")
    print()
    print(f"Hiện có: 218/2000 bình luận. Cần thu thêm ~1800 bình luận.")
    print(f"Mỗi bài viết thường có 10-50 bình luận → cần 40-180 bài viết.")
    print()
    print("Để thu nhanh:")
    print("  - Cuộn feed nhóm, mở từng bài có nhiều bình luận")
    print("  - Ưu tiên bài thảo luận về học tập, thi cử, đăng ký")
    print("  - Tránh bài chỉ có ảnh/sticker mà không có bình luận text")
    print("  - Mỗi lần chạy bookmarklet sẽ xuất bình luận của 1 bài")
    print("  - Hệ thống tự loại trùng khi nhập")


def check_collection_status():
    """Kiểm tra trạng thái thu thập hiện tại."""
    report_path = Path("data/raw/ptit_sources_v1/collection_report.json")
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"\n📊 Trạng thái thu thập:")
        print(f"   Mục tiêu: {report.get('target_comments', 2000)}")
        print(f"   Đã có: {report.get('valid_unique_ids', 0)}")
        print(f"   Bài viết: {report.get('posts', 0)}")
        print(f"   Đạt mục tiêu: {'✅' if report.get('target_reached') else '❌'}")
        remaining = report.get("target_comments", 2000) - report.get("valid_unique_ids", 0)
        if remaining > 0:
            print(f"   Còn thiếu: ~{remaining} bình luận")


if __name__ == "__main__":
    generate_bookmarklet()
    check_collection_status()
