# Hướng dẫn gán nhãn

## Ba nhãn và cách quyết định

Mỗi bình luận nhận đúng một nhãn cảm xúc người viết thể hiện. Đọc cả câu,
xác định đối tượng đánh giá, phủ định, mệnh đề đối lập và lời trích dẫn.
Không suy đoán danh tính; xuất hiện trong nhóm PTIT chưa chứng minh người viết là sinh viên PTIT.

| Nhãn | Tiêu chí |
|---|---|
| positive | Khen, hài lòng, biết ơn, ủng hộ hoặc động viên rõ |
| neutral | Hỏi/cung cấp thông tin, không có sắc thái tích cực hay tiêu cực rõ |
| negative | Chê, bất mãn, thất vọng hoặc phàn nàn rõ |

Câu khó hiểu không mặc định neutral. Nếu nhiều sắc thái, chọn phần trội theo
trọng tâm/kết luận; nếu chưa đủ căn cứ thì giữ unresolved và ghi lý do.
Thông tin về sự việc xấu hoặc cảm xúc được trích từ người khác chưa chắc là cảm xúc người viết.

Các ví dụ sau tự viết để giải thích quy tắc, không phải dữ liệu thực nghiệm:

| Trường hợp | Cách xử lý |
|---|---|
| “Thầy giảng rất dễ hiểu” | positive: khen rõ |
| “Bao giờ có lịch thi?” | neutral: hỏi thông tin |
| “Mình chưa hài lòng với lịch học” | negative: phủ định hài lòng |
| “Không biết lịch thi” | Không suy negative chỉ vì có “không” |
| “Không tệ” | Xét toàn câu; có thể tích cực nhẹ |
| Câu cười/emoji đơn lẻ | Có thể vui hoặc mỉa mai; thiếu ngữ cảnh thì unresolved |
| Khen một vế, chê một vế | Chọn sắc thái trội nếu rõ; nếu không thì unresolved |

Từ tục không tự quyết định negative. Câu hỏi có thể là chất vấn/phàn nàn.
Mỉa mai, tiếng lóng và câu bỏ lửng cần rà kỹ; không gán nhãn bằng từ khóa.
Khó gán nhãn không phải lý do xóa bình luận. Lỗi nguồn/không có nội dung
được xử lý chất lượng riêng với lý do loại.

## Nhãn AI hiện tại

Bộ ban đầu có 2.600 nhãn `label_source=ai`, `annotation_status=ai_labeled`;
chưa có nhãn con người được xác nhận. 194 unresolved không vào bộ chia.
Test AI chỉ đánh giá mức phù hợp với nhãn AI, chưa thay thế nhãn người độc lập.
AI dùng text bình luận; đợt này chưa có văn bản bài viết/bình luận cha.
Không tưởng tượng ngữ cảnh. Người rà chỉ dùng ngữ cảnh thật được phép truy cập
và ghi nguồn tại context_used.

Xem [hợp đồng dữ liệu](data_contract.md) để kiểm tra bộ hiện tại và vị trí ZIP hồ sơ.
Không đổi metadata AI thành human_confirmed bằng script.

## Rà độc lập và ghép nhãn người

Khi bắt đầu đợt rà người, người 1 lấy hồ sơ phân công từ ZIP sang thư mục làm
việc riêng. Hồ sơ cũ có annotation_batches_ai_v1/annotator_1.csv đến annotator_4.csv,
base.csv và assignment_manifest.json. Kiểm tra hash và đường dẫn dữ liệu trong
manifest trước khi dùng; không ghi đè bộ AI đang khóa.

Mỗi ID có primary, một số có cross_check/common; 40 mẫu chung cho cả bốn.
File thành viên không hiển thị gợi ý AI. Đọc và lưu nhãn độc lập trước thảo luận.
Giữ nguyên id/post_id/text/role và mọi dòng.
Sau khi chính người rà đọc và đồng ý, điền label, annotation_status=human_confirmed,
reviewer=member_1..member_4, context_used và notes cho câu khó.
Câu chưa rõ giữ label trống/pending_review; sao chép đề xuất không phải xác nhận.

Ghép bằng [annotation_review.py](../src/ptit_sentiment/data/annotation_review.py),
thay đường dẫn hồ sơ bằng thư mục thực tế đã chuẩn bị:

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m ptit_sentiment.data.annotation_review merge --assignment-dir <thu_muc_phan_cong> --output-dir <thu_muc_ket_qua_moi>
```

Công cụ kiểm tra ID/nhãn/reviewer, nội dung, vai trò, dòng thiếu và xác nhận tường minh.
Đọc validation_issues.csv, needs_review.csv và disagreements.csv.
Chỉ các ID đủ xác nhận hợp lệ và đồng thuận mới thành nhãn người; không tự lấy đa số.
Nếu bất đồng, thảo luận rồi lập file phân xử với final_label, annotation_status=human_confirmed,
reviewer, reason; ghép lại bằng --resolutions vào output mới.
Độ đồng thuận/Kappa chỉ tính từ nhãn độc lập thật trước phân xử.
Phát hành phiên bản dữ liệu/bộ chia mới khi chốt nhãn người, giữ dấu vết nhãn AI cũ.
