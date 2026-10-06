# ptit-sentiment-analysis

**Đưa lên GitHub và bàn giao nhóm:** xem [hướng dẫn cụ thể](docs/github_handoff.md).
Dữ liệu thật nhận qua gói riêng; GitHub chứa code, tài liệu và dữ liệu minh họa.
Đã gom bản bàn giao local tại `handoff/PTIT_BAN_GIAO_20261007/`;
xem [cấu trúc và phần việc từng người](docs/team_handoff.md).

**Bộ nhãn AI ban đầu — 06/10/2026:** xem [kiểm toán và hướng dẫn kiểm tra](docs/annotation_review.md).
Đã đọc toàn bộ 2.856 bình luận qua kiểm tra cấu trúc từ 2.864 dòng gốc:
2.600 nhãn AI (positive 362, neutral 1.800, negative 438), 194 unresolved, 70 bị loại.
Nhãn có `label_source=ai`, `annotation_status=ai_labeled`; nhãn người xác nhận thật: 0.
Đã tạo bộ chia `data/splits/`: train 1.801, validation 409, test 390, seed 42;
không giao ID/bài/nội dung hoặc cặp gần trùng đã nhận diện. Test AI chưa thay thế đánh giá bằng nhãn người độc lập.
Đã dọn các bản cũ `assignments_v1/confirmed_v1/real_v1` ngày 07/10/2026;
bản khôi phục nằm trong `outputs/archives/data_cleanup_20261007.zip`.
Data đã thu gọn: chỉ giữ bảng nhãn, ba tập chia và manifest.json. Hồ sơ nguồn/gán nhãn ở outputs/archives/data_minimal_20261007.zip.
Không thu thập lại hoặc huấn luyện trong nhiệm vụ gán nhãn/chia dữ liệu.

**Đề tài: Xây dựng hệ thống phân loại cảm xúc bình luận của sinh viên PTIT trên các nhóm Facebook.**

Dự án cho bốn người: ba nhãn positive/neutral/negative,
BoW/TF-IDF × MultinomialNB/LinearSVC, baseline nhãn phổ biến và PhoBERT tùy chọn.
Code, tài liệu học và báo cáo có dẫn chứng output; không thu thập Facebook tự động.

**Dữ liệu tests/fixtures là tổng hợp để thử code, không phải bình luận Facebook thật.
Chỉ trình bày điểm số ở mục kiểm tra minh họa, không coi là kết quả trên dữ liệu PTIT thật.**


## Dữ liệu và bộ chia hiện tại

Nguồn chính: `data/raw/ptit_sources_v1/comments.csv`, 2.864 dòng từ 453 bài;
2.377 dòng ptit_2k5 và 487 dòng ptit_group_584397217391365.
Dữ liệu đã gán: `data/labeled/labeled_comments.csv`; câu chưa xác định:
hồ sơ unresolved, kiểm tra mẫu và phân công người đã được lưu trong ZIP khôi phục riêng ngoài data.
Lệnh ghép nhãn người và quy tắc sửa/xác nhận nằm trong docs/annotation_review.md.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python scripts/verify_ai_dataset.py
& $python -m ptit_sentiment.data.validate --input data/labeled/labeled_comments.csv
# Chỉ tái tạo khi cần; thư mục đích phải chưa có bộ chia:
& $python -m ptit_sentiment.data.ai_dataset --directory data/labeled --output-dir data/splits/ai_initial_v2 --seed 42
```


Hiện data chỉ có:

```text
data/labeled/labeled_comments.csv
data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv
data/splits/manifest.json
```

manifest.json giữ nguồn nhãn AI, phiên bản, seed/hash và ràng buộc chống rò rỉ.
Các CSV vẫn giữ nguyên nội dung và metadata nguồn. Không tự chia lại hoặc
đổi nhãn khi chạy các mô hình. Để xem nguồn/batch gán nhãn cũ, khôi phục ZIP
riêng; không cần các hồ sơ đó để train/evaluate trên bộ đã chốt.

## Cài đặt trên Windows PowerShell

Python 3.10–3.12 (lần bàn giao kiểm tra bằng 3.12.9).
Chạy từ thư mục gốc dự án. Nếu đã có .venv ở workspace bàn giao, có thể dùng luôn python.exe bên dưới;
khi tạo môi trường mới trên máy khác:

~~~powershell
# Mở PowerShell tại thư mục gốc dự án
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
~~~

Cài editable package để mọi lệnh python -m tìm được package src.
Không cần đổi PYTHONPATH. Nếu muốn dùng lệnh python ngắn:

~~~powershell
.\.venv\Scripts\Activate.ps1
python -m ptit_sentiment.data.validate --help
~~~

Nếu Activate.ps1 bị chính sách PowerShell chặn, dùng trực tiếp .venv\Scripts\python.exe
như các ví dụ sau, không cần đổi chính sách máy.
CLI tự đặt UTF-8 cho stdout/stderr. Khi redirect từ PowerShell 5, ưu tiên tùy chọn --output của CLI
để giữ CSV UTF-8.

## Thử toàn bộ luồng truyền thống

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/run_smoke.py
~~~

Smoke tạo tên lần chạy theo thời gian; hoặc chọn --run-name demo_01 chưa tồn tại.
Nó gọi từng CLI validate/split/train/evaluate/compare/predict, kiểm tra lưu/tải pipeline,
bảo toàn text và không giao post_id/id/text.
Artifact tại artifacts/<run-name>/; bộ chia data/splits/<run-name>/;
metric, confusion matrix, prediction, lỗi và log tại outputs/<run-name>/.
commands.json lưu lệnh, exit code, stdout/stderr; verification.json ghi kiểm tra cuối.

Lần đã kiểm chứng trong workspace này: **complete_20261003**.
Ví dụ dùng ngay artifact có sẵn:

~~~powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.predict --model artifacts/complete_20261003/bow_nb.joblib --text "Thầy giảng rất dễ hiểu, mình hài lòng."
~~~

## Chạy từng bước

Chọn một tên mới để giữ dữ liệu/artifact/kết quả cũ:

~~~powershell
$run = "demo_02"
$python = ".\.venv\Scripts\python.exe"
& $python -m ptit_sentiment.data.validate --input tests/fixtures/synthetic.csv --output "outputs/$run/validation.json"
& $python -m ptit_sentiment.data.split --input tests/fixtures/synthetic.csv --output-dir "data/splits/$run" --config configs/classical.yaml --provenance synthetic
& $python -m ptit_sentiment.training.train_classical --splits-dir "data/splits/$run" --config configs/classical.yaml --artifact-dir "artifacts/$run" --output-dir "outputs/$run/training"
~~~

Train chỉ fit train và chọn theo Macro-F1 validation; xuất bốn pipeline và majority_baseline.joblib.
Không tính metric test trong lệnh train. Đánh giá test riêng sau khi chốt cấu hình:

~~~powershell
$models = @("bow_nb", "tfidf_nb", "bow_svm", "tfidf_svm", "majority_baseline")
foreach ($model in $models) {
    & $python -m ptit_sentiment.evaluation.evaluate --model "artifacts/$run/$model.joblib" --splits-dir "data/splits/$run" --output-dir "outputs/$run/evaluation/$model"
}
$metricPaths = $models | ForEach-Object { "outputs/$run/evaluation/$_/metrics.json" }
& $python -m ptit_sentiment.evaluation.compare --inputs $metricPaths --output "outputs/$run/comparison.csv"
~~~

Evaluate mặc định test; có --split validation để phân tích validation, không dùng bảng đó như điểm test.
Mỗi mô hình có metrics.json/.csv, classification_report.csv, per_label.csv, confusion_matrix.csv/.png, predictions.csv, errors.csv.
Confusion matrix hàng thật/cột dự đoán. Macro metric tính trên đúng ba nhãn.
Compare xuất thêm comparison.png, chỉ tổng hợp lần đã chạy và từ chối dữ liệu/bộ chia khác nhau.

~~~powershell
& $python -m ptit_sentiment.predict --model "artifacts/$run/bow_nb.joblib" --text "Mình không hài lòng với lịch học mới."
& $python -m ptit_sentiment.predict --model "artifacts/$run/tfidf_svm.joblib" --input tests/fixtures/new_comments.csv --output "outputs/$run/new_comments.csv"
~~~

CSV dự đoán mới cần id,text, label tùy chọn. Không có label thì true_label rỗng.
NB/baseline xuất xác suất; LinearSVC chỉ nhãn, không coi decision score là xác suất.
CLI báo lỗi nếu artifact thiếu; không huấn luyện lại.
Tệp/thư mục kết quả đã tồn tại sẽ bị từ chối; chọn tên mới, không ghi đè lần chạy cũ.

## Thay bằng dữ liệu thật

1. Tạo CSV mới data/labeled/ptit_comments.csv và nhập bình luận thật
   được phép sử dụng: id,post_id,text,label. Giữ text gốc, ẩn thông tin định danh trước khi bàn giao.
2. Gán nhãn theo docs/labeling_guide.md; câu không rõ vào needs_review riêng, không mặc định neutral.
3. Validate CSV, xử lý trùng/nhãn sai/ô rỗng. Không tự coi synthetic.csv là dữ liệu thật.
4. Chia vào thư mục mới, đổi --input và --provenance real:

~~~powershell
& $python -m ptit_sentiment.data.validate --input data/labeled/labeled_comments.csv
# Bộ AI hiện tại đã chia ở data/splits; không tạo lại khi chưa đổi dữ liệu.
# Với nhãn người: ghép/xác nhận theo docs/annotation_review.md rồi chọn output mới.
~~~

Với bộ AI hiện tại, dùng data/splits cho cả classical và PhoBERT.
Mặc định 70/15/15, seed 42. Tỷ lệ có thể xấp xỉ do chia nhóm; xem manifest.json.
Mỗi nhãn phải xuất hiện ở ít nhất ba post_id khác nhau và mỗi tập phải đủ ba nhãn.
Nếu dữ liệu nhỏ hoặc không tìm được bộ chia hợp lệ, lệnh báo lỗi; không fallback chia dòng.
Điều chỉnh split.ratios/attempts trước thực nghiệm hoặc bổ sung dữ liệu.
Giữ manifest bất biến; dữ liệu thay đổi phải tạo bộ chia/artifact mới.

## PhoBERT riêng

Phụ thuộc torch/transformers/accelerate/py-vncorenlp chỉ cài khi cần:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e ".[phobert]"
java -version
# Thiết lập JAVA_HOME cho phiên PowerShell từ Java đã cài:
$javaSettings = & java -XshowSettings:properties -version 2>&1
$javaHomeLine = $javaSettings | Select-String '^\s*java.home ='
$env:JAVA_HOME = ($javaHomeLine.ToString() -split ' = ',2)[1].Trim()
~~~

Cần Java 1.8+; cấu hình JAVA_HOME/JVM phù hợp với pyjnius.
[tài liệu VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) nêu yêu cầu Java và tài nguyên local.
Chuẩn bị segmenter khi thực sự bắt đầu phần PhoBERT:

~~~powershell
.\.venv\Scripts\python.exe scripts/setup_vncorenlp.py --output-dir tools/vncorenlp
~~~

Lệnh setup tải JAR và hai tệp word segmentation từ repo chính thức, ghi hash tại resources.json;
không cần wget. Có --revision <commit> để ghim nguồn VnCoreNLP.
Đã kiểm chứng VnCoreNLP và PhoBERT thật trong lần smoke một bước; xem docs/verification.md.
Nếu đã có tài nguyên local, chỉ sửa segmenter_dir trong configs/phobert.yaml.

Cấu hình mặc định vinai/phobert-base, max_length 256, 3 epoch; sửa batch/learning rate/use_cpu/fp16
theo máy. revision đã ghim commit checkpoint được kiểm chứng; đổi checkpoint cần kiểm chứng lại.
PhoBERT dùng NFC/khoảng trắng + RDRSegmenter, giữ hoa thường, URL, emoji và phủ định;
không áp cleaning truyền thống. Theo [tác giả PhoBERT](https://github.com/VinAIResearch/PhoBERT),
input phải tách từ trước tokenizer.
Trainer ghim transformers==4.44.2, dùng eval_strategy và chọn checkpoint bằng validation Macro-F1;
xem [API Trainer](https://huggingface.co/docs/transformers/v4.44.2/main_classes/trainer).

~~~powershell
& $python -m ptit_sentiment.training.train_phobert --splits-dir data/splits/real_v1 --config configs/phobert.yaml --artifact-dir artifacts/real_v1/phobert
& $python -m ptit_sentiment.evaluation.evaluate --model artifacts/real_v1/phobert --splits-dir data/splits/real_v1 --output-dir outputs/real_v1/evaluation/phobert
& $python -m ptit_sentiment.predict --model artifacts/real_v1/phobert --text "Mình hài lòng với buổi tư vấn."
~~~

Có thể thêm outputs/real_v1/evaluation/phobert/metrics.json vào --inputs của compare với các classical
cùng real_v1. Inference dùng model/tokenizer local, không tải checkpoint hoặc train lại.
Model/tokenizer, labels.json, training_config.json, metadata.json và log/checkpoint được lưu cùng artifact.
Nếu di chuyển máy, dùng --segmenter-dir <đường dẫn local mới> trong evaluate/predict.

## Cấu trúc và bàn giao

- configs/: tham số chia/preprocessing/vectorizer/lưới và PhoBERT.
- src/ptit_sentiment/: labels, data, preprocessing, features, models, training, evaluation, predict.
- data/raw,labeled,splits/: dữ liệu thật do nhóm bổ sung; tests/fixtures/: tổng hợp kiểm tra.
- artifacts/: mô hình và metadata; outputs/: kết quả thực tế các lần chạy.
- docs/task_assignment.md: đầu vào/đầu ra/điều kiện hoàn thành/câu hỏi cho bốn người.
- docs/data_contract.md: schema, nhãn, split và định dạng giao tiếp.
- docs/labeling_guide.md, methodology.md: hướng dẫn nhãn và lựa chọn phương pháp.
- docs/report_outline.md: khung báo cáo với ô CHƯA ĐO/CHƯA CHẠY THẬT.
- scripts/run_smoke.py, tests/: kiểm tra tích hợp và ràng buộc dữ liệu.
- docs/verification.md: kết quả kiểm chứng, phiên bản và giới hạn.

Text đã xử lý chỉ nằm trong pipeline, không ghi đè CSV gốc.
Vocabulary/IDF chỉ fit train; không dùng test chọn tham số.
Chỉ tải joblib tin cậy của nhóm. Dữ liệu thật/artifact/outputs/venv mặc định không vào Git.

## Tình trạng kiểm chứng (03/10/2026)

Đã chạy luồng classical đầu cuối, baseline, lưu/tải pipeline, đánh giá/tổng hợp, dự đoán câu/CSV,
kiểm tra không giao post_id/id/text và bộ unittest. Kết quả là kiểm tra code trên dữ liệu tổng hợp.
Môi trường bàn giao dùng .venv --system-site-packages và editable install --no-deps --no-build-isolation
để tái dùng thư viện sẵn có; chưa kiểm chứng cài mới hoàn toàn từ Internet.
Import module PhoBERT không tự import torch/transformers/py_vncorenlp; đã kiểm tra syntax/CLI.
Trạng thái PhoBERT chi tiết được cập nhật tại docs/verification.md; smoke một bước không thay thực nghiệm đầy đủ.
Đã chạy PhoBERT một bước trên GPU RTX 3050 Laptop, reload khớp nhãn validation, evaluate/predict thành công.
Artifact: artifacts/phobert_verified_20261003; kết quả: outputs/phobert_verified_20261003.
Người 4 cần fine-tune đầy đủ với dữ liệu thật và cấu hình phù hợp máy.
Không có kết quả thực nghiệm dữ liệu thật hoặc kết luận mô hình tốt nhất trong lần này.

## Báo cáo và tài liệu học

- Báo cáo tiếng Việt `report/report.md`: bản cũ giữ local để truy vết, chưa đưa lên GitHub;
  cần sửa các khẳng định xác nhận người/Kappa và đối chiếu lại phiên bản kết quả.
- [Hướng dẫn học](docs/learning_guide.md): fit/transform/predict, NB/SVM, BoW/TF-IDF tính tay,
  pretrain/fine-tune, độ đo và file tương ứng. Người phụ trách phần 3 đọc mục 3–8, 10–12.
- `report/evidence/`: bằng chứng bản cũ giữ local; không dùng làm kết quả của bộ nhãn AI hiện tại.
- Người 1 điều phối ghép báo cáo; người 3 phụ trách NB/SVM, đánh giá và phân tích lỗi.

## Tái lập và kiểm tra PhoBERT tối thiểu

Tải chỉ checkpoint công khai, không gửi bình luận:
~~~powershell
$python = ".\.venv\Scripts\python.exe"
$env:HF_HOME = Join-Path (Get-Location) ".cache/huggingface"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
& $python scripts/download_phobert.py
& $python -m ptit_sentiment.training.check_phobert --output outputs/phobert_resources.json
& $python -m ptit_sentiment.training.train_phobert --config configs/phobert_smoke.yaml --splits-dir data/splits/complete_20261003 --artifact-dir artifacts/phobert_smoke_new
& $python -m ptit_sentiment.evaluation.evaluate --model artifacts/phobert_smoke_new --splits-dir data/splits/complete_20261003 --output-dir outputs/phobert_smoke_new/evaluation
& $python -m ptit_sentiment.predict --model artifacts/phobert_smoke_new --input tests/fixtures/new_comments.csv --output outputs/phobert_smoke_new/predictions.csv
~~~

Smoke dùng batch=1, max_steps=1, max_length=64, gradient_checkpointing;
cấu hình thực nghiệm configs/phobert.yaml dùng max_length=256.
local_files_only=true trong smoke để tránh tải bất ngờ; tải trước qua script.
Huấn luyện báo thiết bị thực; inference tôn trọng use_cpu đã lưu, nếu không dùng GPU khả dụng.
Truncation bên phải; thống kê train/validation tại artifact/truncation.json,
test trong metrics.json, CSV dự đoán có tệp .truncation.json bên cạnh.
Muốn bắt buộc CPU, đặt training.use_cpu=true; giảm batch khi thiếu VRAM.
Không thay checkpoint bằng mô hình khác khi có lỗi.

Lưu source CSV, manifest, cấu hình, commit code, môi trường và artifact của mỗi lần chạy.
Giữ nguyên split/seed khi so sánh; một phiên bản dữ liệu mới phải có thư mục mới.
Không commit cache/model/output lớn; chỉ bản evidence tổng hợp có chủ đích được lưu với báo cáo.

Xuất bằng chứng nhỏ cho một lần chạy classical mới:
~~~powershell
& $python scripts/export_report_evidence.py --run-dir outputs/complete_20261003 --splits-dir data/splits/complete_20261003 --output-dir report/evidence_new
~~~

Chạy tự động toàn bộ smoke PhoBERT (tự dò JAVA_HOME nếu chưa đặt):
~~~powershell
.\.venv\Scripts\python.exe scripts/run_phobert_smoke.py --splits-dir data/splits/complete_20261003 --artifact-dir artifacts/phobert_smoke_next --output-dir outputs/phobert_smoke_next
~~~
Script kiểm tra lưu/tải bằng validation_reference.json, chạy evaluate validation/test và predict câu/CSV.
Dùng --skip-train chỉ khi đã có artifact và muốn kiểm tra phần còn lại.
Code PhoBERT tắt tích hợp TensorFlow (USE_TF=0), không cần cài tf-keras.
