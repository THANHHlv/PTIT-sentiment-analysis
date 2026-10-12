# Bàn giao người 3 → người 4 — 09/10/2026

Đã train/evaluate bốn mô hình truyền thống và majority baseline bằng CPU trên dữ liệu thật,
nhãn AI ban đầu. Không triển khai/chạy PhoBERT trong nhiệm vụ này. Không thu thập, đổi nhãn hay chia lại.

## Artifact và dữ liệu dùng chung

- Artifact: `artifacts/part3_20261009_cpu/{bow_nb,tfidf_nb,bow_svm,tfidf_svm,majority_baseline}.joblib`.
- Mỗi artifact có `.metadata.json`: ánh xạ nhãn, hash artifact/config/CSV/manifest,
  nguồn nhãn, phiên bản thư viện, cấu hình được chọn, seed, thời gian và chính sách fit.
- Config: `configs/part3.yaml`; log mọi ứng viên: `outputs/part3_20261009_cpu/training/`.
- Kết quả: `outputs/part3_20261009_cpu/evaluation/<model>/`; bảng `comparison.csv`, biểu đồ `comparison.png`.
- Bộ chia: **chính ba CSV ở `data/splits/`**, không lấy bộ cũ trong ZIP hoặc `real_v1`.
- Dataset: `ai_initial_4e93595ef7d1_seed42`, train/validation/test = 1801/409/390, seed 42.
- Manifest SHA-256: `c743878f73ef20ef786f46f46c62511b894a10891c2893de8d75cb753522e4e3`.
- Nhãn duy nhất: `positive:0`, `neutral:1`, `negative:2`, nguồn ở `src/ptit_sentiment/labels.py`.
  `model.classes_` có thể theo thứ tự khác; luôn ánh xạ bằng tên lớp, không gán cột xác suất theo vị trí giả định.

## Giao diện tải và dự đoán

```python
import pandas as pd
from ptit_sentiment.predict import load_artifact, predict_with_artifact

model, metadata = load_artifact("artifacts/part3_20261009_cpu/bow_nb.joblib")
frame = pd.DataFrame([{"id": "new-01", "text": "Mình chưa hài lòng với lịch học"}])
result = predict_with_artifact(model, metadata, frame)
print(result)
# Bốn pipeline cũng hỗ trợ model.predict([text_goc, ...]).
```

Input CSV UTF-8 cần `id,text`, ID đọc như chuỗi, text không rỗng, label không bắt buộc.
CLI kiểm tra thiếu cột, trùng ID/text và nhãn tùy chọn. Pipeline từ chối câu thiếu/rỗng hoặc rỗng sau xử lý.
Đầu ra có `id,text,true_label,predicted_label,model_name`; true_label rỗng nếu không có label.
NB/baseline có thêm xác suất theo tên lớp; SVM chỉ nhãn. Không train lại khi predict.
Baseline dùng `predict_classical(..., baseline=True)` hoặc giao diện chung trên vì input DummyClassifier khác pipeline.

Pipeline đã thực hiện NFC/khoảng trắng/ký tự lỗi, che email/điện thoại/@handle, lowercase,
URL→urltoken, slang, giữ emoji/dấu câu/phủ định, Underthesea, không loại stop words,
vectorizer từ train rồi classifier. **Truyền text gốc**, không gọi preprocess trước.
Chỉ joblib tin cậy của nhóm mới được tải; dùng môi trường tương thích đã lưu trong metadata.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m ptit_sentiment.predict --model artifacts/part3_20261009_cpu/bow_nb.joblib --text "Thầy giảng rất dễ hiểu."
& $python -m ptit_sentiment.predict --model artifacts/part3_20261009_cpu/tfidf_svm.joblib --input tests/fixtures/new_comments.csv --output outputs/new_predictions_part3.csv
```

CSV fixture chỉ kiểm tra đường dự đoán, không phải kết quả thực nghiệm.
Chọn đường dẫn output chưa tồn tại; tất cả lệnh bảo toàn lần chạy cũ.

## Hàm đánh giá chung cho PhoBERT

PhoBERT dùng đường chuẩn bị đầu vào riêng; không dùng vectorizer hoặc xử lý truyền thống ở trên.
Người 4 phải train và chọn checkpoint trên **cùng manifest** trước khi đo test.

```python
from ptit_sentiment.evaluation.metrics import prediction_frame, export_evaluation
from ptit_sentiment.common import file_hash

# frame là test đọc bằng read_csv; predicted là list nhãn theo đúng thứ tự frame.
predictions = prediction_frame(frame, predicted, "phobert")
metrics = export_evaluation("outputs/phobert_new/evaluation", predictions, {
    "model_name": "phobert", "split": "test", "provenance": "real",
    "split_manifest_sha256": file_hash("data/splits/manifest.json"),
    "evaluation_data_sha256": file_hash("data/splits/test.csv"),
    "labeling_policy": "ai_initial", "human_labels_confirmed": False,
})
```

`compute_metrics(y_true, y_pred)` tính Accuracy, per-label Precision/Recall/F1/support,
Macro-Precision/Recall/F1; undefined→0, matrix 3×3 theo labels.py, hàng thật/cột dự đoán.
`export_evaluation` xuất metrics.json/.csv, per_label.csv, classification_report.csv,
confusion_matrix.csv/.png, predictions.csv, errors.csv; thư mục đích phải mới.
Với artifact PhoBERT đã đúng schema, ưu tiên CLI `ptit_sentiment.evaluation.evaluate`
để tự kiểm tra hash manifest và thêm metadata/thống kê truncation.
Compare từ chối khác manifest/test/provenance/số mẫu hoặc trùng tên model.

## Tái lập và bằng chứng

```powershell
.\.venv\Scripts\python.exe scripts/run_part3.py --run-name part3_rerun_01
```

Lệnh chạy audit dữ liệu/NV2, 57 tests, 24 ứng viên, lưu tất cả lựa chọn trước test,
đánh giá năm artifact, compare, kiểm tra predict câu/CSV và lỗi đầu vào/artifact,
đối chiếu serialize trên toàn test. Không refit train+validation.
Quy tắc hòa: ứng viên xuất hiện trước trong YAML thắng khi Macro-F1 bằng nhau.
SVM được bắt ConvergenceWarning, tăng max_iter tối đa 100000 rồi thử lại một lần;
vẫn chưa hội tụ thì dừng, không lưu như kết quả đạt. Lần chính không phát sinh cảnh báo hội tụ.
`commands.json` lưu lệnh/exit code/stdout/stderr/thời gian; `run_context.json` lưu HEAD,
trạng thái Git và hash code; `verification.json` lưu kiểm chứng toàn luồng.

Giới hạn: 2.600 nhãn đều AI; test chỉ đo phù hợp với AI. Neutral chiếm đa số,
có câu ngắn/đa nghĩa và vector 0. Không kết luận toàn dự án hoàn tất khi PhoBERT và
đánh giá nhãn người độc lập còn ngoài lần này. Xem [báo cáo](../report/part3_classical.md),
[phân tích lỗi](../report/part3_error_analysis.md), [tài liệu học](part3_learning.md).
