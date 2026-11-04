# PTIT Sentiment Analysis — Người 3

Phần người 3: **BoW + NB, TF-IDF + NB, BoW + LinearSVC, TF-IDF + LinearSVC**.
Đã train/evaluate thật bằng CPU trong lần part3_final_20261102.
Mô hình chính là TF-IDF + SVM, chọn bằng Macro-F1 validation=0,577487.
Accuracy test=0,717949; Macro-F1 test=0,535355.
[Báo cáo người 3](report/report.md) chứa đầy đủ số liệu, ma trận và phân tích.

Dùng nguyên dữ liệu/tiền xử lý của người 1/2. Train/validation/test=1801/409/390,
dataset ai_initial_4e93595ef7d1_seed42. **Nhãn tham chiếu vẫn là AI**;
test từng được dùng, chưa phải đánh giá độc lập bằng nhãn người.
Excel review nằm riêng trong outputs/label_review/, chưa ghép vào thực nghiệm.
PhoBERT do người 4 thực hiện; chưa có điểm PhoBERT trong kết quả này.

## Cài đặt

Repo GitHub giữ mã nguồn, cấu hình, tài liệu Markdown và fixture tổng hợp trong
`tests/fixtures/`. Dữ liệu thật (`data/`, kể cả manifest chứa ID nguồn), model
(`artifacts/`), kết quả chạy (`outputs/`), gói bàn giao (`handoff/`), bằng chứng
sinh tự động (`report/evidence/`) và PDF/Word xuất báo cáo chỉ lưu nội bộ, được
loại khỏi Git bằng `.gitignore`.

Sau khi clone, nhận gói dữ liệu nội bộ từ thành viên phụ trách và khôi phục đúng
`data/labeled/labeled_comments.csv`, `data/splits/train.csv`, `validation.csv`,
`test.csv`, `manifest.json`. Giữ nguyên bộ chia và metadata; không tự chia lại.
Các lệnh kiểm tra dữ liệu và huấn luyện bên dưới cần các file này. Model không
kèm repo; dùng lệnh tái lập phần 3 để tạo artifact trước khi chạy demo dự đoán.

Mở PowerShell tại gốc dự án; khuyến nghị Python 3.12 để khớp artifact.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
~~~

Nếu đã có .venv thì dùng trực tiếp các lệnh bên dưới, không cần biến $python.
Phiên bản thư viện đã train được lưu trong metadata và requirements của gói bàn giao.

## Demo phần 3

~~~powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.predict --model artifacts/part3_final_20261102/tfidf_svm.joblib --text "Thầy giảng rất dễ hiểu."
.\.venv\Scripts\python.exe -m ptit_sentiment.predict --model artifacts/part3_final_20261102/tfidf_svm.joblib --input tests/fixtures/new_comments.csv --output outputs/demo_part3.csv
~~~

Đổi tfidf_svm.joblib thành bow_nb.joblib, tfidf_nb.joblib hoặc bow_svm.joblib để thử
từng tổ hợp. Nhãn trả về positive/neutral/negative. Pipeline nhận text gốc, không train
lại lúc predict. CSV mới cần id,text, không cần label. Đổi tên output nếu file đã tồn tại.

## Kiểm tra đầu vào và code

~~~powershell
.\.venv\Scripts\python.exe scripts/verify_ai_dataset.py --output outputs/data_check_new.json
.\.venv\Scripts\python.exe -m ptit_sentiment.data.validate --input data/labeled/labeled_comments.csv
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
~~~

Có 68 kiểm thử, gồm kiểm tra output dùng chung với nhãn dự đoán PhoBERT.
Fixture tổng hợp chỉ thử code, không thay thực nghiệm thật.

## Tái lập toàn bộ phần 3

Chọn run-name mới để không ghi đè kết quả đã có. Runner audit dữ liệu/NV2,
train 16 ứng viên, chọn bằng validation, khóa artifact rồi mới evaluate test:

~~~powershell
.\.venv\Scripts\python.exe -B scripts/run_part3.py --run-name part3_reproduce_new --config configs/part3.yaml
~~~

Cấu hình: n-gram (1,1)/(1,2), alpha=0.5/1, C=0.5/1, class_weight=None, seed42.
Mỗi tổ hợp bốn ứng viên; không refit train+validation, không tune theo test.
Baseline majority học lớp neutral chỉ từ train để đối chiếu.

## Train, evaluate, compare riêng

Lệnh train chạy đủ bốn pipeline và baseline, nhận cùng bộ chia:

~~~powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.training.train_classical --splits-dir data/splits --config configs/part3.yaml --artifact-dir artifacts/manual_new --output-dir outputs/manual_new/training
~~~

Sau khi đã chốt lựa chọn bằng validation, đánh giá bốn pipeline:

~~~powershell
foreach ($name in @("bow_nb", "tfidf_nb", "bow_svm", "tfidf_svm")) {
    .\.venv\Scripts\python.exe -m ptit_sentiment.evaluation.evaluate --model "artifacts/manual_new/$name.joblib" --splits-dir data/splits --split test --output-dir "outputs/manual_new/evaluation/$name"
}
.\.venv\Scripts\python.exe -m ptit_sentiment.evaluation.compare --inputs outputs/manual_new/evaluation/bow_nb/metrics.json outputs/manual_new/evaluation/tfidf_nb/metrics.json outputs/manual_new/evaluation/bow_svm/metrics.json outputs/manual_new/evaluation/tfidf_svm/metrics.json --output outputs/manual_new/comparison.csv
~~~

Mỗi mô hình xuất metrics.json, classification_report.csv, confusion_matrix.csv/.png,
predictions.csv và errors.csv; có thêm bảng per-label/nhóm lỗi để đối chiếu.
Confusion matrix: hàng nhãn thật, cột dự đoán, thứ tự positive/neutral/negative.
SVM decision score không phải xác suất.

## Kết quả và bàn giao

| Đường dẫn | Công dụng |
|---|---|
| configs/part3.yaml | Cấu hình hiện hành đúng yêu cầu người 3 |
| src/ptit_sentiment/ | Module dữ liệu, tiền xử lý, mô hình, train/evaluate/predict |
| features.py | API đặc trưng bàn giao người 2 |
| data/ | Labeled CSV, ba tập chia và manifest nguyên bản |
| artifacts/part3_final_20261102/ | Bốn pipeline và baseline, mỗi model có metadata |
| outputs/part3_final_20261102/comparison_core.csv | Bảng so sánh đúng bốn mô hình bắt buộc |
| outputs/part3_final_20261102/evaluation/ | Kết quả test và lỗi từng mô hình |
| outputs/part3_final_20261102/delivery/ | Validation, results.csv, primary_model.json, biên bản gói |
| outputs/part3_final_20261102/error_examples.csv | 25 lỗi có ID thật, dùng nội bộ để đối soát |
| outputs/label_review/ | Hai Excel gán nhãn/review, được giữ riêng |
| outputs/archives/ | Nguồn dữ liệu và bản lịch sử trước khi hoàn thiện |
| report/ | Báo cáo, phân tích lỗi ẩn danh và bằng chứng hiện hành |
| handoff/NV3_20261102.zip | Gói code/artifact/kết quả tổng hợp cho người 4 |
| tests/ | Bảy file kiểm thử và fixture riêng |

Gói dữ liệu nội bộ: handoff/ai_initial_4e93595ef7d1_seed42.zip.
Gói NV3 không chứa CSV bình luận thật, ID gốc, dự đoán/lỗi nguyên văn hoặc Excel nhãn.
Cách kiểm tra ZIP và ghép kết quả PhoBERT: [bàn giao](docs/part3_handoff.md).
Lý thuyết để bảo vệ: [part3_learning.md](docs/part3_learning.md).
[Phương pháp](docs/methodology.md), [phân công](docs/task_assignment.md),
[hợp đồng dữ liệu](docs/data_contract.md), [hướng dẫn nhãn](docs/labeling_guide.md).

## Công dụng các script còn giữ

| Script | Vai trò |
|---|---|
| run_part3.py | Audit, train, khóa lựa chọn, evaluate và kiểm chứng |
| review_part2.py | Kiểm tra tích hợp dữ liệu/đặc trưng của người 2 |
| audit_selected_features.py | Đối chiếu vocabulary/IDF chỉ từ train |
| verify_ai_dataset.py | Kiểm tra nguồn nhãn, hash và chống rò rỉ |
| export_report_evidence.py | Xuất bằng chứng báo cáo từ kết quả thật |
| finalize_part3.py | Kiểm chứng kết quả và đóng gói bàn giao |
| verify_part3_bundle.py | Kiểm tra hash và dự đoán sau giải nén |
| package_team_data.py | Đóng gói cùng bộ dữ liệu cho nhóm |
| download_phobert.py | Người 4 tải checkpoint |
| setup_vncorenlp.py | Người 4 chuẩn bị đầu vào PhoBERT |

PhoBERT có nhóm phụ thuộc riêng: pip install -e ".[phobert]"; Java/VnCoreNLP
và configs/phobert*.yaml được giữ cho người 4. Phần người 3 không huấn luyện mô hình này.
Không chia lại bộ hiện hành. Lệnh split cho dữ liệu mới theo [hợp đồng](docs/data_contract.md);
ví dụ kiểm tra code, xuất riêng bằng fixture:

~~~powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.data.split --input tests/fixtures/synthetic.csv --output-dir outputs/splits_example --config configs/classical.yaml --provenance synthetic
~~~
