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