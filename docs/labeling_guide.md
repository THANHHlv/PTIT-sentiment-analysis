# Hướng dẫn gán nhãn

Ví dụ dưới đây tự viết, không phải bình luận Facebook thu thập.
Gán cảm xúc người viết đối với nội dung (môn học, dịch vụ, sự kiện, trải nghiệm PTIT).
Mỗi bình luận đã chốt có đúng một nhãn.

| Nhãn | Ý nghĩa | Ví dụ |
|---|---|---|
| positive | Khen, hài lòng, cảm ơn, mong đợi tích cực | Thầy giải thích dễ hiểu, học mạng máy tính vui quá. |
| neutral | Thông tin/hỏi đáp không thể hiện đánh giá cảm xúc | Lịch thi cơ sở dữ liệu đăng ở đâu vậy? |
| negative | Chê, thất vọng, bất mãn, lo lắng rõ ràng | Cổng đăng ký học cứ lỗi, mình rất bực. |

1. Đọc cả câu, ngữ cảnh bài viết nếu được phép, emoji, phủ định và tiếng lóng.
2. Chọn sắc thái chiếm ưu thế khi nhiều sắc thái: “Phòng đẹp nhưng mạng chậm quá, rất thất vọng”
   thường negative; “Ban đầu khó nhưng thầy hỗ trợ tốt, mình hài lòng” thường positive.
3. Câu không rõ, thiếu ngữ cảnh, mỉa mai/bất đồng: đưa vào data/raw/needs_review.csv
   với id,post_id,text,reason,annotator_a,annotator_b,final_decision.
   Đây là danh sách làm việc riêng, không vào train/test; không mặc định neutral.
4. Hai người gán độc lập một phần dữ liệu, đối chiếu quy tắc; người thứ ba phân xử nếu cần.
   Chỉ chuyển sang labeled sau khi chốt một nhãn hợp lệ.
5. Rà trùng, nhãn mâu thuẫn và nguồn trước khi khóa bộ chia.

“Không tốt” negative; “chưa có lịch thi” có thể neutral khi chỉ báo thông tin;
“không tệ, rất đáng học” có thể positive. Không chỉ nhìn một từ.
“Đỉnh thật, lại sập cổng đăng ký rồi 🙃” có thể mỉa mai; không tự gán positive.
Chửi đùa, emoji đơn lẻ, nhiều đối tượng hoặc trích dẫn ý kiến người khác cần xét kỹ.

Báo cáo ghi số đã gán/bị loại/cần xem xét, số người gán, tỷ lệ bất đồng và cách xử lý.
Chỉ điền Cohen's kappa nếu đã tính trên nhãn độc lập thật.
Không mô tả dữ liệu tổng hợp kiểm tra code như dữ liệu sinh viên thật.
