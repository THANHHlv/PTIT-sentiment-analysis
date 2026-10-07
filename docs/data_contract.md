# Hợp đồng dữ liệu và giao tiếp

> Cập nhật 07/10/2026: data đã thu gọn, chỉ còn labeled_comments.csv và
> train/validation/test cùng manifest.json chứa metadata nhúng. Hồ sơ gán
> nhãn, dữ liệu thô, unresolved/excluded đã lưu riêng trong
> outputs/archives/data_minimal_20261007.zip. Các đường dẫn hồ sơ/batch trong
> nội dung lịch sử dưới đây chỉ dùng sau khi khôi phục ZIP, không có trong
> data hiện tại. Kiểm tra bộ đang dùng bằng scripts/verify_ai_dataset.py;
> đóng gói bằng scripts/package_team_data.py. Fixture ở tests/fixtures/.

## CSV đầu vào

CSV UTF-8 (chấp nhận BOM), header id,post_id,text,label. Mỗi dòng một bình luận.
Cột đọc dưới dạng chuỗi, giữ số 0 đầu mã; cho phép cột bổ sung để lưu nguồn nội bộ.
id duy nhất; post_id là bài viết nguồn; hai mã không có khoảng trắng đầu/cuối.
text là nội dung gốc không rỗng. label đúng một nhãn, không có khoảng trắng dư.
Nguồn duy nhất: src/ptit_sentiment/labels.py, positive=0, neutral=1, negative=2.
Không tạo mapping riêng trong notebook. Mẫu nhập thật data/labeled/template.csv chỉ có header.
needs_review không đưa vào CSV đã chốt. Không đưa dữ liệu thật/định danh cá nhân lên Git.

Validator từ chối id trùng và text trùng sau NFC, casefold, gộp khoảng trắng.
Khóa so trùng không ghi đè text, không phát hiện được mọi paraphrase.
Người 1 phải rà gần trùng, sao chép và nhãn mâu thuẫn; không tự xóa/chọn nhãn khi gặp trùng.

## Bộ chia bất biến

train.csv, validation.csv, test.csv cùng schema, cùng thư mục manifest.json.
Tỷ lệ mặc định 70/15/15; thực tế xấp xỉ vì nhóm post_id không chia nhỏ được.
Mỗi tập đủ ba nhãn; không giao id/post_id/text key.
Mỗi nhãn cần xuất hiện trong ít nhất ba nhóm (điều kiện cần, chưa đảm bảo tìm được bộ chia).
Nếu không tìm được: bổ sung dữ liệu/nhóm, điều chỉnh tỷ lệ hoặc attempts; không fallback chia dòng.
Manifest chứa SHA256 nguồn/từng CSV, seed, tỷ lệ yêu cầu/thực tế, phân bố nhãn, danh sách id/post_id,
thuật toán và provenance real/synthetic. Khóa bộ chia trước thực nghiệm; không chọn seed theo điểm test.
Train xác minh toàn vẹn cả bộ chia nhưng chỉ fit text/label train, chọn tham số bằng validation;
không tính metric test. Tất cả mô hình dùng cùng manifest.
Evaluate kiểm tra hash manifest; compare từ chối trộn bộ chia/tập/hash dữ liệu.
Sửa dữ liệu phải tạo bộ chia mới và artifact mới.

## Artifact và kết quả

Classical: <model_name>.joblib là pipeline preprocessing → vectorizer → classifier.
<model_name>.metadata.json chứa nhãn, cấu hình, tham số chọn, phiên bản và hash bộ chia.
Baseline là DummyClassifier học nhãn phổ biến nhất từ train, bỏ qua text.
Chỉ tải joblib do nhóm tạo và tin cậy vì pickle/joblib có thể thực thi code.

PhoBERT: thư mục Hugging Face model/tokenizer, labels.json, training_config.json, metadata.json,
checkpoint/log trainer. Quản lý VnCoreNLP local riêng.

CSV đánh giá: id,text,true_label,predicted_label,model_name; text giữ nguyên.
NB/baseline có predict_proba, PhoBERT có softmax thì thêm
probability_positive,probability_neutral,probability_negative theo đúng nhãn.
LinearSVC không có xác suất; không biến decision_function thành probability.
Xác suất chưa hiệu chỉnh không phải độ tin cậy đã kiểm định.
CSV mới cần ít nhất id,text; label tùy chọn; true_label rỗng nếu chưa gán nhãn.

Evaluate xuất metrics.json/.csv, per_label.csv, confusion_matrix.csv/.png,
predictions.csv, errors.csv. JSON có label order, hàng thật/cột dự đoán, support,
Macro-P/R/F1, accuracy, provenance, hash bộ chia/dữ liệu.
Metric không xác định được tính 0 (zero_division=0).
Compare chỉ tổng hợp metrics.json đã chạy, không tự thêm mô hình chưa chạy.

## Bổ sung phiên bản hoàn thiện

classification_report.csv chứa P/R/F1/support từng nhãn và dòng macro_avg;
per_label.csv vẫn giữ để tương thích. Compare xuất thêm PNG cạnh CSV/JSON.
Thông tin nhạy cảm được che trong bản xử lý theo mask_sensitive, không sửa text nguồn.
Regex nhận email, số điện thoại VN và @handle; không đảm bảo ẩn danh tên người/địa chỉ.
Dữ liệu thật và CSV predictions/errors phải được rà soát trước chia sẻ.
PhoBERT lưu truncation.json cho train/validation; metrics.json có trường truncation khi evaluate.
Predict PhoBERT với --output ghi thêm <tên>.truncation.json.

## Hợp đồng nhập và xác nhận dữ liệu thật

Nguồn thô `comments.csv` có thêm source_id/source_url/collected_at, không lưu tên tác giả.
`collection_report.json` truy vết hash, checkpoint, loại/trùng và lỗi nguồn.
Bảng gán nhãn có role/suggested_label/label/confirmed/notes; chỉ label được người xác nhận mới ghép.
`labeling_report.json` status=confirmed, provenance=real và labeled_sha256 phải khớp CSV
khi chia thật. Manifest bộ chia lưu bản báo cáo/hash; train và evaluate xác minh lại chúng.
Các cột gợi ý không đi vào nhãn chuẩn. Xem [quy trình](real_data_workflow.md).
