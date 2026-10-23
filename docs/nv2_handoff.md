# Bàn giao người 2 — tiền xử lý và đặc trưng

Bản hợp nhất sau đối chiếu ngày 12/10/2026. Code NV2 đã tích hợp với phần người 3;
CSV hiện hành khớp hash manifest. Không đổi nhãn hoặc chia lại dữ liệu.

## Đầu vào và đầu ra

Bộ ai_initial_4e93595ef7d1_seed42 có 2.600 bình luận thật mang nhãn AI:
train/validation/test = 1.801/409/390. Nhãn chưa được con người xác nhận độc lập.
Text gốc được giữ trong CSV; pipeline nhận text gốc và xử lý khi fit/predict.

| Thành phần | Mã nguồn | Vai trò |
|---|---|---|
| API tiền xử lý | [preprocess.py](../src/ptit_sentiment/preprocess.py) | Xử lý một câu hoặc chuỗi văn bản |
| Chuẩn hóa | [normalize.py](../src/ptit_sentiment/preprocessing/normalize.py) | NFC, khoảng trắng, URL, thông tin nhạy cảm, tiếng lóng |
| Tách từ | [tokenize.py](../src/ptit_sentiment/preprocessing/tokenize.py) | Underthesea, từ nhiều âm tiết nối underscore |
| Đặc trưng | [vectorizers.py](../src/ptit_sentiment/features/vectorizers.py) | BoW, TF-IDF word và word-char |
| API bàn giao | [features.py](../features.py) | Tạo ma trận train/validation/test và transform câu mới |

## Cấu hình được dùng trong thực nghiệm

[part3_v2.yaml](../configs/part3_v2.yaml) chuẩn hóa NFC/khoảng trắng/ký tự lỗi,
chuyển chữ thường, thay URL, che email/điện thoại/@handle. Chỉ mở rộng ko/khum/hok
thành không; giữ k/dc/đc/mn/sv vì đa nghĩa. Giữ emoji/dấu câu, stop words rỗng;
không loại phủ định. Underthesea tách từ trước vectorizer.

BoW và TF-IDF thử unigram, unigram–bigram; min_df=1, max_df=1.0, max_features=20000.
Vocabulary, DF và IDF chỉ học từ train. Các ma trận là sparse không âm:
unigram 3.569 cột, unigram–bigram 17.841 cột.
Nhánh bổ sung TF-IDF word-char nối hai vector L2 riêng, char_wb (3,5).

[configs/classical.yaml](../configs/classical.yaml) giữ cấu hình NV2 tổng quát;
muốn tái lập điểm báo cáo phải dùng part3_v2 và part3_v2_extended.

## Đối chiếu tích hợp

Đã ghép marker merge ở module mô hình/train, đối chiếu nguồn dữ liệu và khôi phục
byte CSV theo bản chuẩn vì sai khác CRLF/LF. Manifest giữ nguyên;
[.gitattributes](../.gitattributes) bảo vệ byte khi checkout.

Bằng chứng hiện hành: outputs/part3_20261012_nv2_v2/part2_audit.json và
feature_audits/. Các audit kiểm tra vocabulary/IDF chỉ từ train, heldout không
đổi tham số và pipeline tải lại cho điểm validation đúng điểm lựa chọn.
Script [review_part2.py](../scripts/review_part2.py) kiểm tra tích hợp;
[audit_selected_features.py](../scripts/audit_selected_features.py) đối chiếu pipeline đã chọn.

Kết quả và API cho người 4 ở [bàn giao người 3](part3_handoff.md);
phương pháp ở [methodology.md](methodology.md), số liệu ở [báo cáo](../report/report.md).
Bản lịch sử và file chạy thử đã chuyển vào outputs/archives/project_cleanup_20261023.zip.
