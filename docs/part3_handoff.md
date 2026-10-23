# Bàn giao phần người 3 — bản hiện hành

Hoàn thiện và kiểm chứng lại ngày 21/10/2026. Phần người 3 gồm NB, SVM, baseline, đánh giá,
phân tích lỗi và báo cáo truyền thống. Sử dụng mô hình đã khóa của
part3_20261012_nv2_v2; không train/refit lại và không chọn bằng test.

## Đầu vào và trạng thái

2.600 bình luận thật: positive 362, neutral 1800, negative 438.
Train/validation/test = **1801/409/390**, seed 42, chia theo nhóm bài/ràng buộc trùng.
Dataset ai_initial_4e93595ef7d1_seed42.
Manifest SHA-256: c743878f73ef20ef786f46f46c62511b894a10891c2893de8d75cb753522e4e3.
Không đổi nhãn hoặc chia lại.

| Đầu ra | Trạng thái |
|---|---|
| Bốn pipeline bắt buộc | BoW/TF-IDF word × MultinomialNB/LinearSVC đã train |
| Majority baseline | Học neutral từ train, đánh giá cùng test |
| Chọn tham số | 24 ứng viên bắt buộc; Macro-F1 validation, không refit |
| Bổ sung | 48 ứng viên TF-IDF word/word-char + SVM, tên riêng |
| Test và validation | Đủ Accuracy, per-label/macro P/R/F1, support, ma trận và dự đoán/lỗi |
| Phân tích lỗi | 25 trích đoạn thật đã ẩn danh, đối chiếu từng dòng và hash |
| Kiểm thử | 67/67 tests đạt, fixture tách khỏi thực nghiệm |
| Gói bàn giao | handoff/NV3_20261021.zip, kèm hash và verifier sau giải nén |


## Quyết định dùng nhãn AI ngày 21/10/2026

Theo yêu cầu của người dùng, tiếp tục dùng bộ nhãn AI ban đầu cho phần người 3.
Bản điền đủ 390 nhãn: outputs/test_ai_labeled_20261021/test_390_ai_labeled.xlsx.
390 nhãn được lấy đúng từ test.csv hiện hành, không gán lại bằng dự đoán NB/SVM.
Tất cả giữ label_source=ai, annotation_status=ai_labeled.
25 nhãn người dùng nhập vẫn là bản nháp pending_review, giữ riêng trong file gốc
và sheet Nhan_ban_da_nhap; 5 bất đồng được lưu để đối chiếu, không tự nhận là nhãn người xác nhận.
Không đổi nhãn/bộ chia/artifact; điểm số và lựa chọn validation giữ nguyên.

## Kết quả và lựa chọn chính

| Mô hình | Macro-F1 validation | Accuracy test | Macro-F1 test |
|---|---:|---:|---:|
| BoW + NB | **0.616621** | 0.694872 | 0.544067 |
| TF-IDF word + NB | 0.504317 | 0.723077 | 0.466176 |
| BoW + SVM | 0.558472 | 0.710256 | 0.547529 |
| TF-IDF word + SVM | 0.577487 | 0.717949 | 0.535355 |
| Majority baseline | 0.272640 | 0.694872 | 0.273323 |
| TF-IDF word-char + SVM (bổ sung) | 0.605381 | 0.707692 | 0.563401 |

**Mô hình chính: BoW + NB**, chọn từ validation, unigram và alpha=0.1.
Không thay lựa chọn theo thứ hạng test. primary_model.json ghi tên/artifact SHA/metric chọn.
Chi tiết: [báo cáo](../report/report.md).

## Artifact và bằng chứng

- artifacts/part3_20261012_nv2_v2/: bốn pipeline, majority_baseline và metadata.
- artifacts/part3_20261012_nv2_v2/extended/: tfidf_svm_extended.joblib và metadata.
- outputs/part3_20261012_nv2_v2/: training/, training_extended/, feature_audits/,
  all_selection_lock.json, evaluation/ và verification.json của lần train/test.
- outputs/part3_completion_20261021/: validation/ với đầy đủ output của sáu mô hình,
  results.csv, primary_model.json, bundle_manifest.json, completion.json.
- outputs/part3_delivery_20261021/tests.json: log 67 tests và hash code lúc kiểm tra.
- handoff/NV3_20261021.zip: gói riêng cho nhóm.

ZIP chứa code/cấu hình, artifact/metadata, kết quả tổng hợp, báo cáo, log validation
và fixture dự đoán. Không chứa CSV bình luận thật, predictions/errors có text gốc,
ID bài Facebook hoặc hồ sơ gán nhãn. Output chi tiết vẫn đủ trong workspace;
người 4 nhận data/ hiện hành riêng từ người 1.

## Chạy trên máy nhận bàn giao

Giải nén ZIP vào thư mục mới, mở PowerShell tại gốc thư mục đó:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-part3.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
$python = ".\.venv\Scripts\python.exe"
& $python scripts/verify_part3_bundle.py --root .
& $python -m ptit_sentiment.predict --model artifacts/part3_20261012_nv2_v2/bow_nb.joblib --text "Mình chưa hài lòng với lịch học."
& $python -m ptit_sentiment.predict --model artifacts/part3_20261012_nv2_v2/tfidf_svm.joblib --input tests/fixtures/new_comments.csv --output outputs/predict_new.csv
~~~

Verifier kiểm tra hash từng tệp và tải sáu artifact dự đoán fixture,
không cần dữ liệu thật/không train. Môi trường thực nghiệm Python 3.12;
requirements-part3.txt của ZIP và metadata chốt phiên bản thư viện.

Để tái lập train/evaluate, nhận data/ hiện hành rồi chọn run-name mới:

~~~powershell
& $python -m ptit_sentiment.data.validate --input data/labeled/labeled_comments.csv
& $python scripts/run_part3.py --run-name part3_reproduce_new --config configs/part3_v2.yaml --extended-config configs/part3_v2_extended.yaml
~~~

Mặc định train/runner dùng configs/part3_v2.yaml. Không refit train+validation.
SVM chưa hội tụ được thử lại một lần; vẫn lỗi thì dừng. Lệnh bảo toàn output đã có.

## API cho người 4

~~~python
import pandas as pd
from ptit_sentiment.predict import load_artifact, predict_with_artifact

model, metadata = load_artifact("artifacts/part3_20261012_nv2_v2/bow_nb.joblib")
frame = pd.DataFrame([{"id": "new-01", "text": "Thầy giảng rất dễ hiểu."}])
print(predict_with_artifact(model, metadata, frame))
~~~

Truyền **text gốc**, không preprocess trước. CSV cần id,text; label không bắt buộc;
post_id được giữ nếu có. Pipeline gồm tiền xử lý + vectorizer + classifier.
labels.py ánh xạ positive=0, neutral=1, negative=2; không suy diễn thứ tự classes_.
SVM chỉ nhãn, decision score không phải xác suất/độ mạnh cảm xúc.
Baseline dùng API chung vì cách nhận input khác pipeline.

PhoBERT dùng đầu vào riêng, cùng manifest/nhãn. Người 4 tái sử dụng prediction_frame,
compute_metrics, export_evaluation trong [metrics.py](../src/ptit_sentiment/evaluation/metrics.py),
hoặc CLI evaluate. Confusion matrix theo positive/neutral/negative, hàng thật/cột dự đoán.
Compare từ chối khác manifest/test/provenance/số mẫu hoặc trùng tên model.

## Giới hạn cần ghi khi báo cáo

Nhãn đều AI, chưa có nhãn người độc lập; test đã dùng ngày 09/10.
Điểm mới là so sánh bổ sung trên cùng test, chưa xác nhận chất lượng độc lập.
Không tune/đổi nhãn tiếp từ lỗi test này. Emoji keep_and_map không đổi đầu vào ở bộ
hiện tại vì không có emoji thuộc bảng ánh xạ; chưa đo được hiệu quả mapping.
PhoBERT thuộc người 4; chưa được fine-tune trong phần người 3.

Học/bảo vệ: [tài liệu phần 3](part3_learning.md), [25 câu sai](../report/part3_error_analysis.md).
