# Bàn giao người 3 cho người 4

Lần part3_final_20261102 đã chạy bốn tổ hợp BoW/TF-IDF × MultinomialNB/LinearSVC.
Cấu hình ở [part3.yaml](../configs/part3.yaml): n-gram (1,1)/(1,2),
alpha=0.5/1.0, C=0.5/1.0, class_weight=None, seed 42.
Baseline neutral chỉ dùng đối chiếu. Không huấn luyện PhoBERT trong phần người 3.

## Đầu vào và đầu ra

Dataset ai_initial_4e93595ef7d1_seed42, train/validation/test = 1801/409/390.
Manifest SHA-256: c743878f73ef20ef786f46f46c62511b894a10891c2893de8d75cb753522e4e3.
Test SHA-256: afcb8ce871c14ea258fc02a8ec56596a2c7e2e034b88d2437f2cdffdc0184312.

Giữ nhãn AI và toàn bộ bộ chia bàn giao. Hai Excel trong outputs/label_review/
là hồ sơ review riêng, chưa thay test hiện hành. Test từng được dùng trước;
kết quả hiện tại chưa thay nhãn người độc lập.

- artifacts/part3_final_20261102/: bốn pipeline, baseline và metadata.
- outputs/part3_final_20261102/comparison_core.csv: bảng bốn mô hình.
- outputs/part3_final_20261102/evaluation/: metric, ma trận, dự đoán và lỗi trên test.
- outputs/part3_final_20261102/delivery/: validation/, results.csv, primary_model.json
  và biên bản đóng gói.
- outputs/part3_final_20261102/checks/tests.json: kiểm thử và hash code.
- handoff/NV3_20261102.zip: gói code, cấu hình, pipeline và kết quả tổng hợp.
- handoff/ai_initial_4e93595ef7d1_seed42.zip: dữ liệu nhận riêng trong nhóm.

Gói NV3 không chứa bình luận thật, ID gốc, CSV predictions/errors hoặc Excel gán nhãn.
Báo cáo gồm trích đoạn đã ẩn danh. Không đưa dữ liệu riêng của nhóm lên kho công khai.

## Chạy gói sau giải nén

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-part3.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts/verify_part3_bundle.py --root .
.\.venv\Scripts\python.exe -m ptit_sentiment.predict --model artifacts/part3_final_20261102/tfidf_svm.joblib --text "Thầy giảng rất dễ hiểu."
~~~

Verifier đối chiếu từng SHA-256 rồi tải pipeline và dự đoán fixture; không train.
Dùng Python 3.12 và phiên bản requirements trong ZIP để khớp môi trường thực nghiệm.
Model chính được chọn bằng validation, ghi trong delivery/primary_model.json;
không đổi lựa chọn theo thứ hạng test.

## API dự đoán

~~~python
import pandas as pd
from ptit_sentiment.predict import load_artifact, predict_with_artifact

model, metadata = load_artifact("artifacts/part3_final_20261102/tfidf_svm.joblib")
frame = pd.DataFrame([{"id": "new-01", "text": "Thầy giảng rất dễ hiểu."}])
result = predict_with_artifact(model, metadata, frame)
~~~

Truyền text gốc. Pipeline tự thực hiện ba bước; không preprocess lần nữa.
CSV dự đoán mới cần id,text, không cần label. Ánh xạ nhãn:
positive=0, neutral=1, negative=2. SVM decision score không phải xác suất.

## Ghép PhoBERT thành bảng năm mô hình

Người 4 fine-tune, chọn checkpoint bằng Macro-F1 validation rồi đánh giá cùng test.
Với artifact PhoBERT theo package, dùng CLI evaluate chung. Với nhãn dự đoán từ
mã riêng, dùng các hàm thuần sau; lấy metadata từ manifest đã kiểm tra:

~~~python
from pathlib import Path
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.evaluation.metrics import (
    compute_metrics, prediction_frame, export_evaluation,
)

frames, manifest, digest = load_split_set("data/splits")
test = frames["test"]
# predicted_ids và predicted_labels là đầu ra thật của người 4.
if list(predicted_ids) != test["id"].tolist():
    raise ValueError("Dự đoán PhoBERT phải khớp ID/thứ tự của test.")
predictions = prediction_frame(test, predicted_labels, "phobert")
metrics = compute_metrics(test["label"].tolist(), predicted_labels)
export_evaluation("outputs/phobert_new/evaluation", predictions, {
    "model_name": "phobert", "split": "test",
    "provenance": manifest["provenance"],
    "dataset_version": manifest["dataset_version"],
    "labeling_policy": manifest["labeling_policy"],
    "human_labels_confirmed": manifest["human_labels_confirmed"],
    "split_manifest_sha256": digest,
    "evaluation_data_sha256": manifest["splits"]["test"]["sha256"],
})
~~~

predicted_ids/predicted_labels trong ví dụ là dữ liệu cần cung cấp, không phải kết quả đã có.
Không truyền metric tự tính vào metadata. export_evaluation tự tính từ nhãn thật/dự đoán
và xuất đủ sáu file bắt buộc. Confusion matrix theo positive/neutral/negative,
hàng thật/cột dự đoán. Nếu có xác suất, truyền classes đúng thứ tự của mô hình.

Dùng compare --inputs bốn metrics.json của lần hiện hành và metrics.json PhoBERT.
Compare từ chối khác phiên bản test hoặc trùng tên model. Chưa có output PhoBERT
thì giữ bảng bốn mô hình, không tạo hàng điểm giả.

Học/bảo vệ: [part3_learning.md](part3_learning.md).
Số liệu và phân tích: [report.md](../report/report.md), [lỗi](../report/part3_error_analysis.md).
