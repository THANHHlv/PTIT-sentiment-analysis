# Hợp đồng dữ liệu và giao tiếp

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
