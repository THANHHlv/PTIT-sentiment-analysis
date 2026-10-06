# Hướng dẫn gán nhãn cảm xúc bình luận PTIT

> Cập nhật 07/10/2026: data đã thu gọn, chỉ còn labeled_comments.csv và
> train/validation/test cùng manifest.json chứa metadata nhúng. Hồ sơ gán
> nhãn, dữ liệu thô, unresolved/excluded đã lưu riêng trong
> outputs/archives/data_minimal_20261007.zip. Các đường dẫn hồ sơ/batch trong
> nội dung lịch sử dưới đây chỉ dùng sau khi khôi phục ZIP, không có trong
> data hiện tại. Kiểm tra bộ đang dùng bằng scripts/verify_ai_dataset.py;
> đóng gói bằng scripts/package_team_data.py. Fixture ở tests/fixtures/.

Cập nhật 07/10/2026. Hướng dẫn trước được lưu trong bản khôi phục riêng
`outputs/archives/data_cleanup_20261007.zip`, dưới data/labeled/history/.
Thao tác xác nhận và lệnh ghép: [annotation_review.md](annotation_review.md).

Đợt AI ban đầu được người dùng cho phép: đã đọc toàn bộ 2.856 dòng qua kiểm
tra cấu trúc, lưu 2.600 nhãn với `label_source=ai`,
`annotation_status=ai_labeled`. 194 dòng thiếu căn cứ nằm trong unresolved,
không chia tập. Đây chưa phải nhãn chuẩn con người. Test có nhãn AI chỉ đo
mức phù hợp với bộ nhãn đó, chưa thay thế đánh giá độc lập bằng nhãn người.

## Đơn vị và ba nhãn

Mỗi bình luận nhận một nhãn cảm xúc người viết thể hiện với môn học, trường,
dịch vụ, trải nghiệm, người khác hoặc một sự việc. Việc bình luận xuất hiện
trong nhóm PTIT chưa chứng minh người viết là sinh viên PTIT. Không suy đoán
danh tính hoặc quan điểm ngoài câu.

Đọc cả câu; xác định người bộc lộ cảm xúc, đối tượng đánh giá, phủ định, các
mệnh đề đối lập và lời trích dẫn. Cảm xúc trong lời người khác được thuật
lại không tự thành cảm xúc người bình luận. Thông tin về sự việc xấu chưa
chắc là lời phàn nàn.

| Nhãn | Tiêu chí | Ví dụ thật đã bỏ tag/tên/liên hệ | Giải thích |
|---|---|---|---|
| positive | Khen, hài lòng, biết ơn, ủng hộ, động viên hoặc đánh giá tích cực rõ | “Thông tin này bổ ích quá ạ ^^” | Khen ích lợi thông tin |
| positive | Như trên | “Dạ em cảm ơn mọi người ạ” | Biết ơn trực tiếp |
| negative | Bất mãn, chê, thất vọng, phàn nàn hoặc đánh giá tiêu cực rõ | “Ý thức để xe vẫn chán lắm.” | Chê ý thức gửi xe |
| negative | Như trên | “đề khó và testcase rất ảo” | Chê đề và testcase |
| neutral | Chủ yếu hỏi/cung cấp thông tin, không sắc thái tích cực/tiêu cực rõ | “Nếu mình đi làm/thực tập ở công ty ngoài thì có được tính không?” | Hỏi quy định |
| neutral | Như trên | “30 câu trắc nghiệm và 3 bài code nhé em” | Mô tả cấu trúc đề |

Ví dụ lấy từ các dòng đã đọc trong đợt này; đây là minh họa cho đề xuất AI,
chưa có người xác nhận nhãn. Hướng dẫn không đưa ID/permalink của ví dụ;
bản làm việc riêng giữ nguyên text để kiểm tra nguồn.

## Trường hợp biên

**Câu hỏi:** xét mục đích và giọng điệu. Hỏi thủ tục thường neutral; chất vấn
việc bị tính công nợ hoặc không được thông báo có thể negative. Ví dụ thật
“mình chẳng thấy thầy cô nào nói về cái này hết” thiên phàn nàn.
“Cô cho điểm khó kh ạ” chỉ hỏi trải nghiệm, chưa khẳng định cô cho điểm khó.

**Từ tục:** xét nghĩa cả câu. “cứ đè mấy môn toán mà lấy dễ vl ra” thiên tích
cực; “vcl” đơn lẻ thiếu đối tượng, cần xem xét. Không dùng từ tục làm luật
gán negative.

**Emoji/bình luận ngắn:** giữ nguyên. Ví dụ thật “:))))))” chưa đủ phân biệt
vui và mỉa mai; “Tạch” có thể là trả lời kết quả hoặc bộc lộ thất vọng. Thiếu
câu trước thì giữ pending. “Chúc b sớm hồi phục” mang thiện chí rõ. Dòng chỉ
dấu chấm hoặc tag thuần được tách vì không có nội dung; không áp dụng việc
loại này cho emoji mang sắc thái.

**Phủ định:** xác định phần bị phủ định. Minh họa quy tắc, không phải trích
dữ liệu: “không tốt” phủ định đánh giá tốt; “không tệ” thường chấp nhận hoặc
tích cực nhẹ; “chưa hài lòng” negative; “không biết lịch” là thiếu thông tin,
chưa đủ negative. Không gán theo một từ đứng riêng.

**Nhiều sắc thái:** chọn sắc thái trội nếu trọng tâm/kết luận/mức nhấn mạnh
rõ. Ví dụ thật sau bỏ tên: “giảng hơi khó hiểu vì ... mix nhiều TA nhma điểm
... khá cao” pha chê cách giảng và khen điểm; đợt này giữ trống đề xuất, high.
Dấu ba chấm ở ví dụ này chỉ phần rút gọn trong hướng dẫn, không sửa CSV.

**Mỉa mai:** “Cẩn thận điểm cao quá nhé” có thể khen đùa hoặc mỉa cách cho
điểm. Đề xuất positive/high cần người kiểm tra. “đánh giá tốt thì ẩn danh còn
xấu thì ẩn thân chi thuật” cần xem xét, không ép neutral. Emoji cười, cánh
cụt và kéo dài chữ chỉ là dấu hiệu cần chú ý.

**Ngữ cảnh:** bản dữ liệu có mã bài/URL, chưa có văn bản bài hoặc bình luận
cha trong đợt này. AI chỉ dùng bình luận hiện có. Người duyệt chỉ dùng ngữ
cảnh thực được cung cấp/được phép xem và ghi nguồn trong `context_used`;
không tưởng tượng ảnh/video hoặc nguyên nhân. Bình luận cùng bài chưa chứng
minh quan hệ cha–con.

**Quảng cáo/giao dịch:** lời chào dịch vụ trực tiếp, cam kết và tuyển dụng
theo mẫu được tách với lý do. Hỏi mua/đổi đồ cá nhân, ghép nhóm học hoặc review
trải nghiệm vẫn được giữ. Review có giá/inbox nhưng chưa chắc quảng cáo được
đưa high để rà. Khó gán nhãn không phải lý do xóa.

**Thiếu căn cứ:** giữ label trống, `annotation_status=unresolved` trong bộ AI
(hoặc pending_review trong file giao người duyệt) và ghi
lý do. Không biến câu mơ hồ thành neutral. Nếu lỗi nguồn hoặc hoàn toàn không
có nội dung, báo người điều phối quyết định chất lượng riêng.

## Gán độc lập và xác nhận

Bốn file mới `data/labeled/annotation_batches_ai_v1/annotator_1.csv` đến `annotator_4.csv`
không hiển thị đề xuất AI. Mỗi ID có một primary; cross_check/common được
người khác đọc độc lập. Mẫu chung 40 ID có trong cả bốn file. Lưu nhãn riêng
trước thảo luận.

Điền label bằng positive/neutral/negative. Sau khi chính bạn đọc và đồng ý,
đặt `annotation_status=human_confirmed`, `reviewer=member_1` (hoặc mã người
tương ứng), `context_used=comment_only` nếu chỉ đọc câu; notes ghi tình huống
khó. Câu chưa rõ giữ pending. Không sao chép đề xuất rồi tự động xác nhận.
CSV xác nhận là khai báo người duyệt, không phải chữ ký số chứng minh đã đọc.

Công cụ chỉ xuất ID có đủ xác nhận hợp lệ của mọi người được phân công và
đồng thuận, hoặc có phân xử tường minh sau bất đồng. Không tự lấy đa số.
Nhãn thiếu/sai, thiếu reviewer và bất đồng đều chờ xử lý. Dữ liệu đã xác nhận
có thể là tập con; phần còn lại tiếp tục chờ.

Độ đồng thuận tính trên nhãn độc lập thực tế trước phân xử. Bất đồng nhiều
thì cả nhóm rà quy tắc và gán lại. Không tạo bất đồng giả hoặc đặt trước tỷ
lệ đồng thuận. Đợt hiện tại chưa có nhãn người, chưa có độ đồng thuận/kappa.
