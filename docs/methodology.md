# Phương pháp thực nghiệm

## Dữ liệu và bộ chia

Dùng nguyên bộ ai_initial_4e93595ef7d1_seed42: 2.600 bình luận thật mang nhãn AI,
train/validation/test = 1.801/409/390. Chia theo post_id và ràng buộc trùng/gần trùng,
seed 42; tỷ lệ thực tế ghi trong manifest. ID, bài và nội dung chuẩn hóa không giao nhau.
Nhãn AI đo mức phù hợp với bộ nhãn AI, chưa thay đánh giá bằng nhãn người độc lập.

Không chia lại bộ hiện hành. Quy trình với dữ liệu mới: validate → chia theo bài →
fit train → chọn Macro-F1 validation → khóa artifact → evaluate test → phân tích lỗi.
[Hợp đồng dữ liệu](data_contract.md) quy định schema và nguồn nhãn.

## Tiền xử lý và đặc trưng

Cấu hình thực nghiệm: [part3_v2.yaml](../configs/part3_v2.yaml).
Chuẩn hóa NFC/khoảng trắng/ký tự lỗi, lowercase, thay URL và che email/điện thoại/@handle.
Mở rộng ko/khum/hok → không; giữ k/dc/đc/mn/sv vì đa nghĩa. Giữ emoji, dấu câu,
không bỏ stop words và phủ định. Underthesea nối từ nhiều âm tiết bằng underscore.
Text gốc không bị ghi đè. Các từ điển quy tắc không được sửa theo lỗi test.

BoW dùng CountVectorizer; TF-IDF dùng TfidfVectorizer với smooth_idf, chuẩn L2.
Vocabulary và IDF chỉ học train. Thử word n-gram (1,1)/(1,2), min_df=1,
max_df=1.0, max_features=20000. Nhánh bổ sung nối TF-IDF word và char_wb (3,5),
chuẩn L2 từng nhánh, không chuẩn hóa lại sau nối.

## Chọn mô hình truyền thống

Bốn tổ hợp BoW/TF-IDF × MultinomialNB/LinearSVC, mỗi tổ hợp sáu ứng viên:
alpha NB=0.1/0.5/1 hoặc C SVM=0.1/1/10 kết hợp hai word n-gram. Tổng 24 lượt fit.
Baseline học nhãn phổ biến nhất train, hiện là neutral.

[part3_v2_extended.yaml](../configs/part3_v2_extended.yaml) thử riêng TF-IDF + SVM:
2 n-gram × 3 C × 2 emoji × 2 class weights × 2 biểu diễn = 48 ứng viên.
Dùng tên tfidf_svm_extended để phân biệt bốn tổ hợp bắt buộc.
Emoji keep_and_map không đổi đầu vào ở bộ này do không có emoji thuộc bảng ánh xạ;
chưa thể kết luận hiệu quả mapping từ điểm bằng nhau.

Chọn bằng Macro-F1 validation, đủ ba nhãn, zero_division=0; hòa lấy ứng viên đầu
trong YAML. Không refit train+validation. BoW+NB được chọn làm mô hình chính.
Các pipeline đã khóa trước lệnh test của lần 12/10. Test từng được dùng ngày 09/10,
nên lần 12/10 là so sánh bổ sung trên cùng test, chưa phải xác nhận độc lập.
Không chỉnh tham số hoặc nhãn tiếp theo lỗi test này.

## Đánh giá và truy vết

Xuất Accuracy, Precision/Recall/F1/support từng nhãn, Macro-P/R/F1;
confusion matrix hàng nhãn thật, cột dự đoán, thứ tự positive/neutral/negative.
Compare chỉ nhận cùng manifest, test hash, provenance và số mẫu.
SVM decision score không phải xác suất hay độ mạnh cảm xúc.

outputs/part3_20261012_nv2_v2/ giữ lịch sử lựa chọn, metadata, khóa SHA-256,
ma trận, dự đoán, lỗi và audit. outputs/part3_completion_20261021/ giữ kết quả
validation và lựa chọn chính. [Bằng chứng báo cáo](../report/evidence/README.md)
được xuất từ các kết quả này, không chứa text bình luận.
25 trích đoạn lỗi đã ẩn danh là mẫu định tính có chủ đích; tách quan sát và giả thuyết,
không dùng để suy ra tỷ lệ nguyên nhân của cả bộ.

## PhoBERT

Code fine-tune có trong [train_phobert.py](../src/ptit_sentiment/training/train_phobert.py),
nhưng chưa có mô hình PhoBERT huấn luyện trên bộ chia hiện hành.
Checkpoint smoke và mô hình trên bộ chia cũ được lưu trong ZIP dọn dự án,
không đưa điểm cũ vào bảng so sánh hiện tại.

Đầu vào riêng: NFC/khoảng trắng → RDRSegmenter VnCoreNLP → tokenizer PhoBERT
use_fast=False → truncation bên phải → padding động. Giữ hoa thường/URL/slang/emoji;
không áp toàn bộ cleaning của mô hình truyền thống. Giới hạn tối đa 256 token kể cả
special tokens; code ghi thống kê mẫu bị cắt từng tập. Java và VnCoreNLP model cần sẵn.
Chọn checkpoint theo Macro-F1 validation, dùng cùng manifest và ánh xạ nhãn.
Xem [hướng dẫn chạy](../README.md) và [tài liệu học](learning_guide.md).
