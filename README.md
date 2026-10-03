# Phân loại cảm xúc bình luận sinh viên PTIT trên Facebook group

Khung bài tập lớn cho bốn người: ba nhãn positive/neutral/negative,
BoW/TF-IDF × MultinomialNB/LinearSVC, baseline nhãn phổ biến và PhoBERT tùy chọn.
Chỉ có code, tài liệu và khung báo cáo; không thu thập Facebook tự động.

**Dữ liệu data/examples là tổng hợp để thử code, không phải bình luận Facebook thật.
Không dùng điểm số trên dữ liệu này làm kết quả báo cáo.**

## Cài đặt trên Windows PowerShell

Python 3.10–3.12 (lần bàn giao kiểm tra bằng 3.12.9).
Chạy từ thư mục gốc dự án. Nếu đã có .venv ở workspace bàn giao, có thể dùng luôn python.exe bên dưới;
khi tạo môi trường mới trên máy khác:

~~~powershell
cd D:\PYTHON\NLP\BTL1
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

Lần đã kiểm chứng trong workspace này: **smoke_verified_v2**.
Ví dụ dùng ngay artifact có sẵn:

~~~powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.predict --model artifacts/smoke_verified_v2/bow_nb.joblib --text "Thầy giảng rất dễ hiểu, mình hài lòng."
~~~

## Chạy từng bước

Chọn một tên mới để giữ dữ liệu/artifact/kết quả cũ:

~~~powershell
$run = "demo_02"
$python = ".\.venv\Scripts\python.exe"
& $python -m ptit_sentiment.data.validate --input data/examples/synthetic.csv --output "outputs/$run/validation.json"
& $python -m ptit_sentiment.data.split --input data/examples/synthetic.csv --output-dir "data/splits/$run" --config configs/classical.yaml --provenance synthetic
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
Mỗi mô hình có metrics.json/.csv, per_label.csv, confusion_matrix.csv/.png, predictions.csv, errors.csv.
Confusion matrix hàng thật/cột dự đoán. Macro metric tính trên đúng ba nhãn.
Compare chỉ tổng hợp lần đã chạy và từ chối dữ liệu/bộ chia khác nhau.

~~~powershell
& $python -m ptit_sentiment.predict --model "artifacts/$run/bow_nb.joblib" --text "Mình không hài lòng với lịch học mới."
& $python -m ptit_sentiment.predict --model "artifacts/$run/tfidf_svm.joblib" --input data/examples/new_comments.csv --output "outputs/$run/new_comments.csv"
~~~

CSV dự đoán mới cần id,text, label tùy chọn. Không có label thì true_label rỗng.
NB/baseline xuất xác suất; LinearSVC chỉ nhãn, không coi decision score là xác suất.
CLI báo lỗi nếu artifact thiếu; không huấn luyện lại.
Tệp/thư mục kết quả đã tồn tại sẽ bị từ chối; chọn tên mới, không ghi đè lần chạy cũ.

## Thay bằng dữ liệu thật

1. Sao chép data/labeled/template.csv thành data/labeled/ptit_comments.csv và nhập bình luận thật
   được phép sử dụng: id,post_id,text,label. Giữ text gốc, ẩn thông tin định danh trước khi bàn giao.
2. Gán nhãn theo docs/labeling_guide.md; câu không rõ vào needs_review riêng, không mặc định neutral.
3. Validate CSV, xử lý trùng/nhãn sai/ô rỗng. Không tự coi synthetic.csv là dữ liệu thật.
4. Chia vào thư mục mới, đổi --input và --provenance real:

~~~powershell
& $python -m ptit_sentiment.data.validate --input data/labeled/ptit_comments.csv
& $python -m ptit_sentiment.data.split --input data/labeled/ptit_comments.csv --output-dir data/splits/real_v1 --provenance real --config configs/classical.yaml
~~~

Sau đó dùng data/splits/real_v1 cho cả classical và PhoBERT.
Mặc định 70/15/15, seed 42. Tỷ lệ có thể xấp xỉ do chia nhóm; xem manifest.json.
Mỗi nhãn phải xuất hiện ở ít nhất ba post_id khác nhau và mỗi tập phải đủ ba nhãn.
Nếu dữ liệu nhỏ hoặc không tìm được bộ chia hợp lệ, lệnh báo lỗi; không fallback chia dòng.
Điều chỉnh split.ratios/attempts trước thực nghiệm hoặc bổ sung dữ liệu.
Giữ manifest bất biến; dữ liệu thay đổi phải tạo bộ chia/artifact mới.

## PhoBERT riêng — chưa chạy trong lần bàn giao

Phụ thuộc torch/transformers/accelerate/py-vncorenlp chỉ cài khi cần:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e ".[phobert]"
java -version
~~~

Cần Java 1.8+; cấu hình JAVA_HOME/JVM phù hợp với pyjnius.
[tài liệu VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) nêu yêu cầu Java và tài nguyên local.
Chuẩn bị segmenter khi thực sự bắt đầu phần PhoBERT:

~~~powershell
.\.venv\Scripts\python.exe scripts/setup_vncorenlp.py --output-dir tools/vncorenlp
~~~

Lệnh setup tải JAR và hai tệp word segmentation từ repo chính thức, ghi hash tại resources.json;
không cần wget. Có --revision <commit> để ghim nguồn VnCoreNLP.
Lệnh này và việc tải checkpoint **chưa được chạy** trong lần tạo khung.
Nếu đã có tài nguyên local, chỉ sửa segmenter_dir trong configs/phobert.yaml.

Cấu hình mặc định vinai/phobert-base, max_length 256, 3 epoch; sửa batch/learning rate/use_cpu/fp16
theo máy. Ghim revision bằng commit checkpoint trước thực nghiệm chính thức.
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
- data/raw,labeled,splits/: dữ liệu thật do nhóm bổ sung; data/examples/: tổng hợp kiểm tra.
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

## Tình trạng kiểm chứng

Đã chạy luồng classical đầu cuối, baseline, lưu/tải pipeline, đánh giá/tổng hợp, dự đoán câu/CSV,
kiểm tra không giao post_id/id/text và bộ unittest. Kết quả là kiểm tra code trên dữ liệu tổng hợp.
Môi trường bàn giao dùng .venv --system-site-packages và editable install --no-deps --no-build-isolation
để tái dùng thư viện sẵn có; chưa kiểm chứng cài mới hoàn toàn từ Internet.
Import module PhoBERT không tự import torch/transformers/py_vncorenlp; đã kiểm tra syntax/CLI.
**Chưa tải checkpoint/VnCoreNLP, chưa khởi tạo JVM thật, chưa fine-tune/evaluate/predict PhoBERT thật.**
Người 4 cần xác nhận tương thích Java/pyjnius/torch trên máy và chạy với dữ liệu thật.
Không có kết quả thực nghiệm dữ liệu thật hoặc kết luận mô hình tốt nhất trong lần này.
