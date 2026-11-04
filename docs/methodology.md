# Phương pháp thực nghiệm người 3

Lần hiện hành: part3_final_20261102; cấu hình [part3.yaml](../configs/part3.yaml).
Không đổi dữ liệu, nhãn, bộ chia, quy tắc tiền xử lý hoặc vectorizer của người 2.

## Dữ liệu và tránh rò rỉ

Dùng dataset ai_initial_4e93595ef7d1_seed42, seed 42: train 1.801, validation 409,
test 390; cả ba tập đủ positive/neutral/negative. Chia theo bài và các ràng buộc
trùng/gần trùng đã nhúng trong manifest. Validator kiểm tra hash, ID, post_id,
nội dung chuẩn hóa và phân bố nhãn trước khi chạy.

Tất cả nhãn hiện hành vẫn ai/ai_labeled; chưa có nhãn người xác nhận trong manifest.
Excel review được giữ riêng ở outputs/label_review/, không ghép vào lần chạy này.
Test đã được sử dụng trong các lần trước; đây là thực nghiệm bổ sung trên test đã biết,
không gọi là đánh giá độc lập trên dữ liệu mới hoặc nhãn người.

## Pipeline và lựa chọn

Text gốc → TextPreprocessor của người 2 → CountVectorizer/TfidfVectorizer → NB/LinearSVC.
BoW là số đếm; TF-IDF dùng smooth IDF và chuẩn L2. Vocabulary/DF/IDF chỉ học train.
Validation, test và câu mới chỉ transform/predict. Tiền xử lý theo quy tắc không học nhãn
được cache cho cùng chế độ emoji; mỗi ứng viên có vectorizer/classifier riêng.

Bốn tổ hợp, mỗi tổ hợp bốn ứng viên: word n-gram (1,1)/(1,2) ×
alpha=0.5/1.0 hoặc C=0.5/1.0. Tổng 16 ứng viên. class_weight=None được chốt trước;
không mở lưới sau khi xem test. TF-IDF word, không dùng nhánh word-char trong lần này.
Giữ min_df=1, max_df=1.0, max_features=20000 và cấu hình tiền xử lý bàn giao.

Chọn Macro-F1 validation; hòa lấy cấu hình đầu YAML. Không refit train+validation.
Lưu pipeline/metadata, kiểm tra dự đoán trước/sau joblib và điểm validation từ text gốc.
Khóa SHA-256 mọi artifact trước evaluate test. Baseline dự đoán lớp phổ biến nhất
chỉ học từ train, để đối chiếu với bốn thuật toán bắt buộc.

## Đánh giá và truy vết

Accuracy, P/R/F1/support từng nhãn và Macro-P/R/F1 dùng đủ ba nhãn,
zero_division=0. Confusion matrix có hàng nhãn thật, cột dự đoán, thứ tự
positive/neutral/negative. Macro-F1 là trung bình ba F1, không phải F1 của Macro-P/R.
SVM không xuất xác suất. Metadata không được ghi đè chỉ số tính từ dự đoán.

Các kết quả chỉ được ghép khi cùng manifest, hash CSV đánh giá, split, provenance
và số mẫu. Người 4 có thể dùng trực tiếp compute_metrics, prediction_frame,
export_evaluation trong [metrics.py](../src/ptit_sentiment/evaluation/metrics.py)
cho đầu ra PhoBERT mà không cần artifact sklearn.

Log nằm trong outputs/part3_final_20261102/: training/, feature_audits/,
all_selection_lock.json, evaluation/, verification.json; delivery/ giữ kết quả
validation và gói bàn giao được kiểm tra sau giải nén.
[Báo cáo](../report/report.md) chỉ dùng số liệu của lần này.

## Phân tích lỗi và giới hạn

Chọn mô hình chính bằng validation trước test. Ví dụ lỗi được đọc từ errors.csv,
trích nguyên văn và loại thông tin nhận diện; mã E01–E25 liên kết đến ID thật trong
file local error_examples.csv. Tách quan sát với giả thuyết chưa kiểm định.
Không dùng lỗi để sửa nhãn hoặc tuning tiếp. Các nhóm lỗi tự động có thể giao nhau.

Người 4 tự fine-tune PhoBERT với đầu vào riêng và cùng manifest. Không có điểm
PhoBERT trong bảng hiện hành; cần nhận artifact/output thật của họ trước khi so sánh.
Nhãn AI, mất cân bằng lớp, một bộ chia và test đã dùng là các giới hạn bắt buộc ghi.
