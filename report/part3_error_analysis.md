# Phân tích 25 dự đoán sai của TF-IDF + SVM

Mô hình được chọn bằng validation (Macro-F1=0,577487), không chọn theo test.
Có 110/390 dự đoán khác nhãn tham chiếu AI. Mỗi trích đoạn dưới đây là nguyên văn
từ errors.csv, đã bỏ tên người ở đầu câu khi cần. E01–E25 là mã phân tích.
Bảng local outputs/part3_final_20261102/error_examples.csv chứa thêm ID thật,
nhãn tham chiếu, dự đoán và nhận xét; không đưa ID nguồn vào gói bàn giao.
Ánh xạ/hash nằm trong error_example_links_local.json.

Mẫu được chọn có chủ đích để bao phủ sáu hướng nhầm, không đại diện cho tỷ lệ
nguyên nhân của toàn bộ lỗi. Nhãn AI cũng có thể sai hoặc thiếu ngữ cảnh.
Quan sát mô tả câu; giả thuyết cần kiểm tra thêm, không dùng để tune/sửa nhãn test.

## Nhãn AI negative → neutral: 37 lỗi, 5 ví dụ

| Mã | Nội dung trích nguyên văn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E01 | Vài câu đề k rõ ràng và bị lỗi testcase | Nêu đề không rõ và lỗi testcase, dùng viết tắt k. | Từ viết tắt và từ nội dung kỹ thuật có thể làm tín hiệu chê yếu; cần xem trọng số mới xác nhận. |
| E02 | khê | Chỉ một từ, thiếu đối tượng được nhận xét. | Thiếu ngữ cảnh hoặc từ hiếm có thể khiến mô hình nghiêng neutral; nhãn AI cũng cần rà. |
| E03 | Thầy khó vs cho điểm thấp lắm bn | Có lời chê độ khó/điểm thấp và viết tắt vs/bn. | Biểu diễn token không thể hiện đầy đủ quan hệ giữa đối tượng và lời chê. |
| E04 | Đây phải nhóm của chuẩn đoán bệnh đâu ... | Phản bác bằng cấu trúc phải ... đâu. | N-gram unigram có thể bỏ mất phạm vi phủ định và ý phản bác. |
| E05 | chuột nhiều | Hai từ mô tả hiện tượng, không nêu đối tượng/ngữ cảnh. | Mô hình có thể xem là thông tin; chưa thể khẳng định nhãn AI đúng khi thiếu ngữ cảnh. |

## Nhãn AI negative → positive: 7 lỗi, 5 ví dụ

| Mã | Nội dung trích nguyên văn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E06 | thi hết mình điểm hết hồn nha | Hai vế chơi chữ, có hết mình và hết hồn. | Sắc thái mỉa mai hoặc thất vọng có thể bị tín hiệu ở vế đầu che lấp. |
| E07 | làm bài thư giãn, điểm vượt mức pickleball | Câu dùng so sánh gây cười về điểm và có thư giãn. | Từ tích cực có thể chi phối trong khi ý đánh giá gián tiếp không được biểu diễn. |
| E08 | review sáng nay trời lạnh, đói do chưa ăn sáng =)) | Tả lạnh/đói cùng biểu tượng cười ASCII. | Sắc thái pha trộn và emoticon có thể gây nhầm; cần ngữ cảnh để kiểm tra nhãn. |
| E09 | Anh này đc A+ mà hay nói phét quá à | Trộn thành tích A+ với chê nói phét. | Unigram có thể không mô tả đúng quan hệ tương phản qua từ mà. |
| E10 | đã vote 1 sao, xin cảm ơn. | Có đánh giá 1 sao và lời cảm ơn trong cùng câu. | Lời lịch sự có thể lấn tín hiệu chê; đây là giả thuyết về mỉa mai. |

## Nhãn AI neutral → negative: 20 lỗi, 4 ví dụ

| Mã | Nội dung trích nguyên văn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E11 | Bạn thi lại à, bạn thi lại thì connect mình với. Mình cũng thi lại nên về việc bảo lưu mình đang hơi không hiểu | Hỏi kết nối và thủ tục, có thi lại và không hiểu. | Các token gắn với khó khăn có thể kéo dự đoán sang negative dù câu chủ yếu hỏi thông tin. |
| E12 | giỏi-khá-tb lên hết như nhau | So sánh các mức học lực bằng câu rất ngắn. | Thiếu ngữ cảnh có thể khiến quan sát bình thường bị hiểu là phàn nàn. |
| E13 | Ide nào cũng được mà | Trả lời công cụ IDE nào cũng được. | Từ kỹ thuật/ngữ cảnh phản hồi ít tín hiệu, dự đoán có thể do liên hệ thống kê của token. |
| E14 | Dev C | Chỉ có tên công cụ. | Tên công cụ không tự biểu đạt cảm xúc; mô hình có thể dựa vào liên hệ trong train. |

## Nhãn AI neutral → positive: 11 lỗi, 3 ví dụ

| Mã | Nội dung trích nguyên văn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E15 | Qua môn thì dễ nhưng để điểm cao thì khó | Có cả dễ và khó, nối bằng nhưng. | Unigram có thể ưu tiên từ tích cực, chưa mô tả quan hệ giữa hai mục tiêu. |
| E16 | Gửi thông tin A1 | Yêu cầu thông tin ngắn, không có lời khen rõ. | Ít token và liên hệ thống kê của A1 có thể ảnh hưởng kết quả. |
| E17 | 10đ | Một số điểm ngắn, không có ngữ cảnh. | Mô hình có thể liên hệ điểm cao với positive; nhãn cần ngữ cảnh hội thoại. |

## Nhãn AI positive → negative: 8 lỗi, 4 ví dụ

| Mã | Nội dung trích nguyên văn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E18 | Triết thì dễ | Nhận xét môn dễ, có từ tích cực trực tiếp. | Từ chỉ môn học có thể lấn từ dễ; cần kiểm tra tham số để khẳng định. |
| E19 | may quá toàn đi bộ | Có may quá thể hiện nhẹ nhõm. | Từ nội dung có thể lấn tín hiệu cảm xúc ngắn. |
| E20 | chỉ muốn ở đây luôn :))) | Muốn ở lại, kèm biểu tượng cười ASCII. | Tín hiệu ASCII hoặc ý thích gián tiếp có thể được học chưa đủ. |
| E21 | Công nhận TADA đi rẻ vch | Công nhận giá rẻ, có viết tắt nhấn mạnh. | Slang và tên dịch vụ có thể làm mô hình bỏ lỡ lời khen. |

## Nhãn AI positive → neutral: 27 lỗi, 4 ví dụ

| Mã | Nội dung trích nguyên văn | Quan sát | Giả thuyết cần kiểm tra |
|---|---|---|---|
| E22 | pldc đc trắc nghiệm là êm r | Nhận xét thuận lợi về hình thức thi bằng nhiều viết tắt. | Slang map bàn giao không mở rộng các từ đa nghĩa, nên có thể thiếu tín hiệu tích cực. |
| E23 | chỉ biết là mấy bạn học chăm, điểm học bổng cao lắm | Có lời khen học chăm, điểm cao trong câu kể. | Nội dung thông tin có thể lấn phần khen trong biểu diễn từ. |
| E24 | Không gì là không thể e ạ | Động viên bằng hai phủ định. | Unigram khó biểu diễn phủ định kép và ý khích lệ. |
| E25 | Khong sao đâu em , Mình còn nhiều nháp mà :< | An ủi, dùng Khong không dấu và biểu tượng ASCII. | Biến thể không dấu và lời an ủi gián tiếp có thể không trùng tín hiệu trong train. |

## Nhận xét toàn bộ test

Positive→neutral có 27 câu và negative→neutral có 37 câu, chiếm 64/110 lỗi
(58,18%). Nhầm hai lớp thiểu số sang neutral là hướng nổi bật. Câu ngắn, viết tắt,
phủ định, tương phản và mỉa mai xuất hiện trong mẫu, nhưng chưa gán taxonomy cho
toàn bộ 110 lỗi nên không suy ra tỷ lệ từng nguyên nhân.

Nhóm error_groups.csv là dấu hiệu tự động, có giao nhau, không phải kết luận
nguyên nhân. Bước nghiên cứu tiếp theo cần test mới có nhãn người và phân tích
trọng số/thí nghiệm kiểm soát; không điều chỉnh mô hình trên test đã đọc này.
Xem [báo cáo](report.md) và [bàn giao](../docs/part3_handoff.md).
