# Hợp đồng dữ liệu

## Bộ đang sử dụng

Phiên bản: `ai_initial_4e93595ef7d1_seed42`, seed 42. Dữ liệu bình luận thật,
nhãn AI ban đầu được người dùng cho phép; chưa có nhãn người xác nhận.
Test AI chỉ đo mức phù hợp với nhãn AI, chưa thay thế đánh giá bằng nhãn người độc lập.

`data/` chỉ giữ:

- `labeled/labeled_comments.csv`: 2.600 bình luận có nhãn.
- `splits/train.csv`, `validation.csv`, `test.csv` và `manifest.json`.

| Tập | Tổng | positive | neutral | negative |
|---|---:|---:|---:|---:|
| train | 1.801 | 254 | 1.246 | 301 |
| validation | 409 | 58 | 283 | 68 |
| test | 390 | 50 | 271 | 69 |
| Tổng | 2.600 | 362 | 1.800 | 438 |

Nguồn thô, 194 unresolved, 70 bị loại và hồ sơ gán nhãn nằm trong
`outputs/archives/data_minimal_20261007.zip`; không tự đưa lại vào data.
Manifest nhúng metadata nguồn/nhãn, hash, seed và ràng buộc chống rò rỉ.
Dữ liệu tổng hợp để thử code nằm ở `tests/fixtures/`.

## CSV và nhãn

CSV UTF-8, chấp nhận BOM; mỗi dòng một bình luận. Đọc các cột dạng chuỗi để giữ mã dài/số 0 đầu.

| Cột | Yêu cầu |
|---|---|
| id | Duy nhất, không rỗng, không có khoảng trắng thừa |
| post_id | Mã bài viết thật, không rỗng; không tạo mã giả |
| text | Nội dung gốc không rỗng, không ghi đè bằng text đã xử lý |
| label | positive, neutral hoặc negative cho mọi dòng huấn luyện |
| label_source / annotation_status | Bộ hiện tại: ai / ai_labeled |
| source_id / source_url / collected_at | Giữ metadata nguồn hiện có |
| reason / review_priority | Lý do và mức ưu tiên rà soát; không phải xác suất |

Ánh xạ duy nhất tại [labels.py](../src/ptit_sentiment/labels.py): positive=0, neutral=1, negative=2.
Câu chưa đủ căn cứ giữ label trống trong hồ sơ unresolved, không đưa vào bộ chia.
Validator từ chối ID trùng và text trùng sau NFC/casefold/gộp khoảng trắng;
khóa so trùng không sửa text và không phát hiện mọi cách diễn đạt lại.

Khi mở CSV bằng Excel, nhập qua Data → From Text/CSV, chọn UTF-8 và đặt mọi cột
thành Text trước Load; tránh đổi mã dài hoặc thực thi nội dung như công thức.

## Bộ chia và kiểm tra

Mọi mô hình dùng cùng bộ chia đã khóa. Tỷ lệ mục tiêu 70/15/15, tỷ lệ thực tế
xấp xỉ do chia nguyên nhóm bài viết. Bộ hiện tại còn nối nhóm theo quan hệ
trùng lịch sử và các cặp gần trùng đã nhận diện.
Không giao ID, post_id, nội dung chuẩn hóa hoặc thành phần nhóm giữa các tập;
mỗi tập đủ ba nhãn. Không chia lại trên từng máy hoặc chọn seed theo điểm test.
Đổi dữ liệu/nhãn phải phát hành bộ chia và artifact phiên bản mới.

Chạy từ gốc dự án sau khi cài package theo [README](../README.md):

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python scripts/verify_ai_dataset.py
& $python -m ptit_sentiment.data.validate --input data/labeled/labeled_comments.csv
# Đích phải chưa tồn tại; gói chỉ dùng để chia sẻ nội bộ:
& $python scripts/package_team_data.py --output handoff/minimal_ai_dataset_new.zip
```

Kiểm tra bộ tối giản đối chiếu CSV, hash và metadata nhúng; không đọc lại nguồn thô đã lưu ZIP.
Mỗi thành viên nhận cùng gói riêng, sao chép data/ vào gốc clone rồi kiểm tra.
Không commit bình luận thật, dữ liệu nhận diện, cookie, cache, model hoặc output lớn.

## Artifact và kết quả

Classical: `<model_name>.joblib` lưu pipeline tiền xử lý → vectorizer → classifier;
`<model_name>.metadata.json` lưu nhãn, cấu hình, tham số, phiên bản và hash bộ chia.
Chỉ tải joblib do nhóm tạo và tin cậy.
PhoBERT lưu model/tokenizer, labels.json, training_config.json, metadata.json,
truncation.json và log/checkpoint; tài nguyên VnCoreNLP quản lý riêng.

Evaluate xuất metrics.json/.csv, classification_report.csv, per_label.csv,
confusion_matrix.csv/.png, predictions.csv và errors.csv.
CSV dự đoán đánh giá có id,text,true_label,predicted_label,model_name.
NB/baseline và PhoBERT có thể xuất xác suất theo tên nhãn;
LinearSVC không có xác suất. Xác suất nhãn không phải độ mạnh cảm xúc.

CSV dự đoán mới cần id,text; label tùy chọn. Predict không fit lại mô hình.
Evaluate yêu cầu nhãn; compare chỉ nhận kết quả cùng phiên bản bộ chia/tập/hash.
Confusion matrix theo positive, neutral, negative; hàng thật, cột dự đoán.
Text trong predictions/errors vẫn có thể chứa thông tin cá nhân; rà soát trước chia sẻ.
