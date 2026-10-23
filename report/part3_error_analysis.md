# Phân tích 25 câu sai — phiên bản NV2 mới, 12/10/2026

Mô hình phân tích là **BoW + NB**, được chọn từ validation (Macro-F1=0,616621),
không chọn bằng test. Có 119/390 dự đoán khác nhãn AI. Toàn bộ ví dụ dưới đây
là trích đoạn thật đã đối chiếu trong `errors.csv`; đã bỏ tên người và không đưa ID Facebook.
Trích đoạn có thể cắt đầu/cuối, không viết lại lời bình luận. Tên chứng chỉ/dịch vụ còn lại
không phải tên người. E01–E25 là mã minh họa mới, không phải ID dữ liệu.

Mẫu được chọn có chủ đích để bao phủ sáu hướng nhầm, không ngẫu nhiên và không đại diện
cho tần suất từng nguyên nhân. Quan sát dựa vào câu; giả thuyết chưa được kiểm định bằng
trọng số từng token hay thí nghiệm can thiệp. Nhãn AI có thể sai hoặc thiếu ngữ cảnh;
không đổi nhãn/tune mô hình sau phân tích này.

Bằng chứng: `outputs/part3_20261012_nv2_v2/anonymized_error_examples.csv`;
`error_example_links_local.json` giữ ánh xạ/hash để đối soát local, không đưa vào báo cáo/kho công khai.

## Nhãn AI negative → neutral (tổng 34 câu; minh họa 5 câu)

| Mã | Trích đoạn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E01 | khê | Câu chỉ có một từ, thiếu ngữ cảnh. | Từ hiếm/ngắn có thể không đủ tín hiệu; nhãn AI cần rà độc lập. |
| E02 | Thầy khó vs cho điểm thấp lắm bn | Có cụm “khó”, “điểm thấp”, viết tắt vs/bn. | Slang và cách kết hợp từ có thể làm tín hiệu chê yếu đi. |
| E03 | Đây phải nhóm của chuẩn đoán bệnh đâu ... | Câu phản bác, không dùng dạng “không” trực tiếp. | BoW khó biểu diễn phạm vi phủ định và ý phê bình. |
| E04 | Sinh viên vẫn đi từ tầng G lên ầm ầm. Xếp hàng đợi trên tầng 1 thang lên từ tầng G chật cứng coi có tức không. | Có mô tả sự việc lẫn câu hỏi “có tức không”. | Từ thông tin có thể lấn tín hiệu bất mãn; đây là giả thuyết. |
| E05 | chuột nhiều | Hai từ mô tả hiện tượng, nhãn AI là negative. | Thiếu đối tượng/ngữ cảnh khiến cảm xúc khó xác định. |

## Nhãn AI negative → positive (tổng 6 câu; minh họa 5 câu)

| Mã | Trích đoạn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E06 | bài kiểu ý làm căng lém e ơi | Có slang “lém”, câu than khó. | Biến thể viết có thể không trùng từ cảm xúc đã học. |
| E07 | thi hết mình điểm hết hồn nha | Cấu trúc hai vế, sắc thái gây bất ngờ ở vế sau. | Token tích cực ở vế đầu có thể dẫn tới positive. |
| E08 | Anh này đc A+ mà hay nói phét quá à | Trộn thành tích A+ và chê “nói phét”. | BoW không thể hiện quan hệ tương phản giữa hai vế. |
| E09 | t buồn lắm | Có từ cảm xúc trực tiếp “buồn”. | Một từ đúng nghĩa vẫn có thể thua prior hoặc các trọng số khác; cần xem tham số mới kết luận. |
| E10 | đã vote 1 sao, xin cảm ơn. | Có “1 sao” và lời cảm ơn trong cùng câu. | Ý mỉa mai có thể bị đọc theo token lịch sự/tích cực. |

## Nhãn AI neutral → negative (tổng 30 câu; minh họa 4 câu)

| Mã | Trích đoạn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E11 | Chỉ đi trong ds cty liên kết vs trường or thực tập tại trường | Nêu phạm vi lựa chọn, nhiều viết tắt. | Các từ ràng buộc có thể gần những câu phàn nàn đã học. |
| E12 | Bạn thi lại à, bạn thi lại thì connect mình với. Mình cũng thi lại nên về việc bảo lưu mình đang hơi không hiểu | Hỏi kết nối và nêu việc không hiểu thủ tục. | Từ “thi lại”, “không hiểu” có thể gợi negative dù AI gán neutral. |
| E13 | Chẳng khó khăn gì , học hết kì 1 | Phủ định khó khăn trong câu giải thích. | BoW không diễn giải đầy đủ “chẳng ... gì”; bigram chưa được chọn. |
| E14 | Có lẽ b nên hỏi trong nhóm tai nghe | Đề nghị hỏi ở nhóm khác. | Lời góp ý có thể gần cách phản bác; nhãn cũng phụ thuộc ngữ cảnh hội thoại. |

## Nhãn AI neutral → positive (tổng 17 câu; minh họa 3 câu)

| Mã | Trích đoạn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E15 | Qua môn thì dễ nhưng để điểm cao thì khó | Câu so sánh hai mục tiêu, có cả dễ và khó. | Token “dễ”, “điểm cao” có thể lấn vế tương phản. |
| E16 | Gửi thông tin A1 | Yêu cầu cung cấp thông tin, không có khen rõ. | Liên hệ thống kê của A1 hoặc các từ ít ỏi có thể chi phối nhãn. |
| E17 | Nàng thơ có tội tình j hã | Câu hỏi mang tính hội thoại, viết tắt j. | Thiếu ngữ cảnh và sắc thái hỏi tu từ; chưa đủ để khẳng định nhãn AI đúng. |

## Nhãn AI positive → negative (tổng 8 câu; minh họa 4 câu)

| Mã | Trích đoạn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E18 | may quá toàn đi bộ | Có từ “may quá” thể hiện nhẹ nhõm. | Từ nội dung có thể lấn tín hiệu cảm xúc ngắn. |
| E19 | chỉ muốn ở đây luôn :))) | Muốn ở lại và có emoticon văn bản. | Emoticon ASCII không được xử lý như emoji Unicode; cần kiểm tra vocabulary trước kết luận. |
| E20 | Chỉ một từ thôi "Vuýpppp" | Từ nhấn mạnh bằng kéo dài ký tự. | Chưa chuẩn hóa kéo dài chữ nên dễ thành token hiếm. |
| E21 | Công nhận TADA đi rẻ vch | Lời công nhận giá rẻ, kèm viết tắt nhấn mạnh. | Slang có thể gắn với nhiều sắc thái; BoW thiếu quan hệ lời khen. |

## Nhãn AI positive → neutral (tổng 24 câu; minh họa 4 câu)

| Mã | Trích đoạn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E22 | cảm ơn b | Lời cảm ơn ngắn, viết tắt b. | Có thể thiếu đủ đặc trưng để thắng prior neutral. |
| E23 | pldc đc trắc nghiệm là êm r | Nhận xét thuận lợi về hình thức thi, nhiều viết tắt. | Từ biểu đạt tích cực không được mở rộng tự động trong slang map an toàn. |
| E24 | Aptis thì dễ nhất, IELTS thì sau đi làm ổn nhất nha | So sánh tích cực về hai chứng chỉ. | Token thông tin/tên riêng có thể chi phối hơn “dễ”, “ổn”. |
| E25 | chỉ biết là mấy bạn học chăm, điểm học bổng cao lắm | Có lời khen chăm học và điểm cao. | Câu thông tin xen khen có thể gần neutral trong dữ liệu train. |

## Nhận xét từ toàn bộ test

BoW+NB nhầm 24/50 positive và 34/69 negative thành neutral. Hai hướng này chiếm
58/119 lỗi, khoảng 48,7%. Có câu ngắn, slang, phủ định gián tiếp, mỉa mai và trộn sắc thái
trong mẫu đã đọc; chưa gán taxonomy thủ công cho toàn bộ 119 lỗi nên không báo tỷ lệ
từng nguyên nhân. `error_groups.csv` chỉ dùng dấu hiệu tự động và các nhóm có giao nhau.

Nhánh word-char có 0 vector 0 ở validation, nhưng không giải quyết mọi câu đa nghĩa hoặc
thiếu ngữ cảnh. Điểm test của nó cao hơn TF-IDF word + SVM về Macro-F1, song đây là
so sánh bổ sung trên test đã dùng trước đó, không chứng minh quan hệ nhân quả hoặc ưu thế phổ quát.

Bước tiếp theo của nhóm là rà nhãn độc lập ở dữ liệu mới, thêm ngữ cảnh khi được phép,
và đánh giá mô hình đã chốt trên test mới mang nhãn người. Không tiếp tục dùng các lỗi
ở đây để chỉnh tham số rồi gọi điểm của chính test này là đánh giá độc lập.

Xem [báo cáo thực nghiệm](report.md) và [bàn giao](../docs/part3_handoff.md).
