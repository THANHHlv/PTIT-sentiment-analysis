# Phân tích 25 câu sai — phần 3

Lần chạy: `part3_20261009_cpu`, dataset `ai_initial_4e93595ef7d1_seed42`.
Phân tích **BoW+NB được chọn vì đứng đầu validation**, không chọn lại mô hình theo test.
Mô hình sai 119/390 câu. Chọn có chủ đích 25 câu để bao phủ cả sáu hướng nhầm lớp;
đây không phải mẫu ngẫu nhiên để ước lượng tỷ lệ loại lỗi. Toàn bộ 119 câu có ở
`outputs/part3_20261009_cpu/evaluation/bow_nb/errors.csv`.

Trích đoạn giữ cách viết gốc, dấu `…` chỉ phần lược bỏ, `[tên đã ẩn]` thay tên người.
Không đưa ID Facebook hoặc tên người vào tài liệu này. Bảng đối chiếu ID/dòng local:
`outputs/part3_20261009_cpu/error_analysis_25.csv`.
“Nhãn thật” dưới đây là **nhãn AI đã khóa**, chưa được người xác nhận độc lập.
Quan sát là nội dung và kết quả thực tế; cột giả thuyết chưa phải giải thích nhân quả đã chứng minh.
Không sửa nhãn, tham số hoặc bộ chia từ những lỗi này.

| Mã | Trích đoạn đã ẩn thông tin nhận diện | Nhãn AI → dự đoán | Quan sát | Giả thuyết nguyên nhân |
|---|---|---|---|---|
| E01 | Thầy khó vs cho điểm thấp lắm bn | negative → neutral | Có phàn nàn về mức điểm và độ khó. | Từ viết tắt `vs`, `bn` và ngữ cảnh đánh giá giảng viên có thể chưa được biểu diễn đủ. |
| E02 | Đây phải nhóm của chuẩn đoán bệnh đâu ... | negative → neutral | Dạng câu hỏi tu từ phản đối nội dung đăng. | Unigram không biểu diễn quan hệ `phải … đâu` và ý phê phán. |
| E03 | chuột nhiều | negative → neutral | Câu chỉ có hai từ, không nêu đối tượng/bối cảnh. | Ít tín hiệu đánh giá trực tiếp; prior neutral có thể lấn át. Nhãn cũng cần được người rà theo bài gốc. |
| E04 | Gẫy gần hết b ạ | negative → neutral | Có từ chỉ kết quả xấu, kèm viết tắt. | Cách viết `Gẫy` và câu thiếu đối tượng có thể khiến đặc trưng ít hữu ích. |
| E05 | Ôn hết lms là đc tầm 40% rồi, nhưng nhớ lưu ý là lms sai khá nh | negative → neutral | Vừa hướng dẫn ôn vừa cảnh báo LMS sai. | Tín hiệu thông tin và bất mãn trộn lẫn; unigram không xác định sắc thái chiếm ưu thế. |
| E06 | Nộp bài xong hoảng luôn mà | negative → neutral | Có biểu hiện lo sợ sau nộp bài. | Từ `hoảng` có thể chưa có trọng số đủ mạnh so với các từ thủ tục học tập. |
| E07 | thi hết mình điểm hết hồn nha | negative → positive | Câu chơi chữ về điểm thi. | Mẫu `hết mình` có thể giống lời động viên; mô hình không hiểu phép đối `điểm hết hồn`. |
| E08 | Anh này đc A+ mà hay nói phét quá à | negative → positive | Có điểm A+ tích cực và chê `nói phét`. | Tín hiệu khen/điểm tốt có thể lấn át mệnh đề chê sau `mà`. |
| E09 | đã vote 1 sao, xin cảm ơn. | negative → positive | Chấm 1 sao đi cùng lời cảm ơn. | Không hiểu thang sao và sắc thái mỉa; `cảm ơn` có thể nghiêng về positive. |
| E10 | Chỉ đi trong ds cty liên kết vs trường or thực tập tại trường | neutral → negative | Trả lời điều kiện nơi thực tập bằng nhiều từ viết tắt. | Từ hạn chế `chỉ` hoặc từ liên quan trường có thể gắn với negative trong train. |
| E11 | … Mình cũng thi lại nên về việc bảo lưu mình đang hơi không hiểu | neutral → negative | Hỏi/trao đổi bảo lưu và thi lại. | `thi lại`, `không hiểu` có thể giống than phiền dù nhãn AI xem là trao đổi thông tin. |
| E12 | Chẳng khó khăn gì , học hết kì 1 … | neutral → negative | Phủ định khó khăn, sau đó giải thích lịch học. | Giữ từ phủ định chưa đủ: unigram vẫn cộng tín hiệu của `khó khăn` độc lập. |
| E13 | Có lẽ b nên hỏi trong nhóm tai nghe | neutral → negative | Đề nghị hỏi ở nhóm chuyên biệt. | Cách diễn đạt hướng người hỏi ra nhóm khác có thể giống lời bác bỏ trong train; cần ngữ cảnh để phân biệt. |
| E14 | … đừng dại mà phi xe lên 2 tầng nha … | neutral → negative | Chỉ dẫn gửi xe có lời cảnh báo. | Từ cảnh báo mang sắc thái âm có thể khiến lời khuyên bị xem là phàn nàn. |
| E15 | Qua môn thì dễ nhưng để điểm cao thì khó | neutral → positive | Nhận xét cân bằng giữa qua môn và điểm cao. | Hai vế trái chiều; tín hiệu `dễ` có thể thắng khi mô hình không xét quan hệ `nhưng`. |
| E16 | Gửi thông tin A1 | neutral → positive | Yêu cầu thông tin rất ngắn. | Cụm hoặc token hiếm có thể chịu ảnh hưởng phân bố từ ở train; chưa đủ căn cứ xác định token gây lỗi. |
| E17 | 10đ | neutral → positive | Chỉ một biểu thức điểm số. | Điểm cao có thể gắn với lời khen trong train, trong khi nhãn cần ngữ cảnh bài viết. |
| E18 | Của các bạn học chất lượng cao nhaa | neutral → positive | Thông tin về chương trình chất lượng cao. | Cụm tên chương trình chứa các từ đánh giá `chất lượng`, `cao`, dễ bị hiểu thành khen. |
| E19 | Chỉ một từ thôi "Vuýpppp" | positive → negative | Dùng biến thể kéo dài/không chuẩn để khen. | Không có quy tắc chuẩn hóa biến thể này; mô hình có thể thiếu tín hiệu tích cực hữu ích. |
| E20 | Công nhận TADA đi rẻ vch | positive → negative | Khen giá rẻ nhưng có tiếng lóng mạnh. | Từ lóng có thể gắn với negative trong train và lấn át `rẻ`/`công nhận`. |
| E21 | Rất đánh giá cao những mong muốn và câu hỏi của [tên đã ẩn] ha. | positive → negative | Có lời ghi nhận rõ `đánh giá cao`. | Cách viết lịch sự/độ dài và các từ nội dung có thể khác mẫu khen train; chưa kiểm tra đóng góp từng từ. |
| E22 | Aptis thì dễ nhất, IELTS thì sau đi làm ổn nhất nha | positive → neutral | Khuyến nghị tích cực giữa hai chứng chỉ. | Tên chứng chỉ và cấu trúc tư vấn có thể khiến mô hình nghiêng về trao đổi thông tin. |
| E23 | chỉ biết là mấy bạn học chăm, điểm học bổng cao lắm | positive → neutral | Khen học chăm và điểm học bổng cao. | Tín hiệu thành tích chen giữa nhiều từ thông tin học tập; có thể bị prior neutral lấn át. |
| E24 | Không gì là không thể e ạ | positive → neutral | Câu động viên có phủ định kép. | Unigram không kết hợp hai phủ định để suy ra ý khích lệ. |
| E25 | Thừa sức bro ơi | positive → neutral | Động viên ngắn, có tiếng Anh khẩu ngữ. | Cụm động viên/ngữ cảnh gọi bạn có thể ít mẫu hoặc thiếu trọng số đủ mạnh. |

## Quan sát định lượng trên toàn test

BoW+NB có 24 positive và 33 negative bị đoán neutral; đồng thời 16 neutral bị đoán positive,
31 neutral bị đoán negative, 9 positive bị đoán negative và 6 negative bị đoán positive.
Tổng là 119 lỗi. Như vậy mô hình không chỉ bỏ sót cảm xúc mà còn gán cảm xúc cho câu thông tin.
Recall positive=0,34, negative≈0,4348; kết quả này cho thấy hạn chế trên hai lớp nhỏ.

Những giả thuyết từ bảng gồm thiếu ngữ cảnh bài/đoạn hội thoại, tiếng lóng/biến thể viết,
phủ định phức hợp, mỉa mai và câu trộn sắc thái. Đây là hướng cho **nghiên cứu sau**;
không chỉnh pipeline sau khi đã mở test của lần chạy này.
Nhãn AI ở câu ngắn/đa nghĩa cũng có thể sai; cần người rà độc lập trước khi kết luận lỗi thuộc mô hình.
