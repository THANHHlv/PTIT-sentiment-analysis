# PTIT Sentiment Analysis

**Xây dựng hệ thống phân loại cảm xúc bình luận của sinh viên PTIT trên các nhóm Facebook.**

Dự án xây dựng luồng xử lý từ nhập bình luận, kiểm tra và gán nhãn dữ liệu,
chia tập, tiền xử lý tiếng Việt đến huấn luyện, đánh giá và dự đoán.
Package Python: `ptit_sentiment`. Mỗi bình luận nhận một trong ba nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| `positive` | Khen ngợi, hài lòng, ủng hộ |
| `neutral` | Thông tin hoặc hỏi đáp không có sắc thái rõ |
| `negative` | Bất mãn, chê trách, thất vọng |

Câu mơ hồ cần rà soát, không tự quy thành `neutral`. Phạm vi sản phẩm gồm
CLI, mô hình và báo cáo tiếng Việt; không yêu cầu giao diện web hoặc cloud.

## Các thành phần và trạng thái

| Thành phần | Nội dung và trạng thái hiện tại |
|---|---|
| Dữ liệu | Bộ ban đầu gồm 2.600 bình luận thật, nhãn AI; có bộ chia và manifest truy vết |
| Tiền xử lý | Chuẩn hóa Unicode, khoảng trắng, URL, tiếng lóng và emoji theo cấu hình; tách từ tiếng Việt, giữ phủ định |
| Đặc trưng | BoW và TF-IDF, unigram/unigram–bigram; vectorizer chỉ học từ train |
| Mô hình truyền thống | Đã huấn luyện và đánh giá BoW + NB, TF-IDF + NB, BoW + SVM tuyến tính, TF-IDF + SVM tuyến tính |
| Baseline | Dự đoán lớp phổ biến nhất, học lớp từ train |
| PhoBERT | Có code fine-tune, cấu hình, kiểm tra môi trường và luồng lưu/tải; chưa có kết quả huấn luyện đầy đủ trong báo cáo hiện hành |
| Đánh giá và dự đoán | CLI dùng chung cho pipeline truyền thống và artifact PhoBERT; xuất độ đo, ma trận nhầm lẫn, dự đoán và lỗi |
| Tài liệu | Hợp đồng dữ liệu, hướng dẫn nhãn, phương pháp, phân công, tài liệu học và báo cáo |

Dự án chưa hoàn tất toàn bộ thực nghiệm khi chưa có kết quả PhoBERT đầy đủ.
Nhãn hiện hành có `label_source=ai`, `annotation_status=ai_labeled`, chưa được
xác nhận độc lập bởi người gán nhãn. Test đã từng được sử dụng; kết quả hiện tại
đo mức phù hợp với bộ nhãn AI, chưa thay thế đánh giá bằng nhãn người độc lập.

## Cấu trúc dự án

```text
configs/                  Cấu hình dữ liệu, đặc trưng và huấn luyện
src/ptit_sentiment/
  data/                   Nhập, kiểm tra, gán nhãn và chia dữ liệu
  preprocessing/          Chuẩn hóa và tách từ; đầu vào riêng cho PhoBERT
  features/               BoW và TF-IDF
  models/                 Mô hình truyền thống và PhoBERT
  training/               Huấn luyện, chọn tham số/checkpoint
  evaluation/             Độ đo, đánh giá và so sánh
  predict.py              Dự đoán một câu hoặc CSV
scripts/                  Kiểm chứng, chuẩn bị tài nguyên và đóng gói
features.py               API đặc trưng dùng chung
tests/                   Kiểm thử và fixture tổng hợp
docs/                    Tài liệu kỹ thuật và phân công
report/                  Báo cáo và phân tích lỗi
data/                    Dữ liệu nội bộ, không đưa lên Git
artifacts/               Model và metadata sinh sau huấn luyện, không đưa lên Git
outputs/                 Kết quả thực nghiệm nội bộ, không đưa lên Git
```

## Cài đặt

Dùng PowerShell tại gốc dự án. Package hỗ trợ Python 3.10–3.12;
khuyến nghị Python 3.12 để khớp môi trường artifact hiện hành.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Nhóm phụ thuộc PhoBERT cài riêng theo phần hướng dẫn bên dưới.
Phiên bản thư viện của mỗi lần huấn luyện được lưu trong metadata artifact.

## Chuẩn bị và kiểm tra dữ liệu

CSV UTF-8 có các cột `id`, `post_id`, `text`, `label`; đọc ID dạng chuỗi và
giữ nguyên text gốc. Dữ liệu huấn luyện phải có đủ nhãn hợp lệ, metadata nguồn
và nguồn nhãn. Xem [hợp đồng dữ liệu](docs/data_contract.md) và
[hướng dẫn gán nhãn](docs/labeling_guide.md).

Repo chỉ giữ mã nguồn, cấu hình, tài liệu và fixture tổng hợp.
Sau khi clone, nhận bộ dữ liệu nội bộ từ thành viên phụ trách và khôi phục:

```text
data/labeled/labeled_comments.csv
data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv
data/splits/manifest.json
```

Giữ nguyên bộ chia và metadata của phiên bản `ai_initial_4e93595ef7d1_seed42`:
train/validation/test = **1.801/409/390** dòng. Toàn bộ dữ liệu có 362 positive,
1.800 neutral và 438 negative. Chia theo nhóm bài và ràng buộc nội dung trùng,
seed 42; tỷ lệ thực tế gần mục tiêu 70/15/15. Không tự chia lại bộ hiện hành.
Raw và hồ sơ gán nhãn cũ lưu riêng trong archive nội bộ, không đưa lại vào `data/`.

```powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.data.validate --input data/labeled/labeled_comments.csv
.\.venv\Scripts\python.exe scripts/verify_ai_dataset.py --output outputs/data_check_new.json
```

Với dữ liệu mới, thực hiện luồng gán nhãn và kiểm tra theo hợp đồng trước khi
chia tập. Có thể thử CLI chia tập bằng fixture tổng hợp, xuất vào thư mục riêng:

```powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.data.split --input tests/fixtures/synthetic.csv --output-dir outputs/splits_example --config configs/classical.yaml --provenance synthetic
```

Fixture chỉ kiểm tra code, không thay dữ liệu thực nghiệm. Cấu hình thu thập
không phải lệnh tự động truy cập Facebook; cần nguồn và phương thức truy cập
được phép. Không cần thu thập lại để tái lập bộ hiện hành.

## Huấn luyện mô hình truyền thống

Bốn tổ hợp dùng cùng bộ chia: BoW + Multinomial NB, TF-IDF + Multinomial NB,
BoW + LinearSVC và TF-IDF + LinearSVC. Baseline học lớp phổ biến nhất từ train.
Pipeline lưu đầy đủ tiền xử lý, vectorizer và classifier bằng joblib.

```powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.training.train_classical --splits-dir data/splits --config configs/part3.yaml --artifact-dir artifacts/classical_new --output-dir outputs/classical_new/training
```

`configs/part3.yaml` là cấu hình thực nghiệm truyền thống hiện hành: n-gram
(1,1)/(1,2), alpha NB = 0.5/1, C SVM = 0.5/1, seed 42. Chọn bằng Macro-F1
validation, không refit train+validation. `configs/classical.yaml` cung cấp
cấu hình chung với lưới rộng hơn; thay cấu hình tạo thành một lần thực nghiệm mới.

Có thể tái lập trọn luồng kiểm tra dữ liệu/đặc trưng, huấn luyện 16 ứng viên,
khóa lựa chọn, đánh giá và kiểm chứng bằng runner:

```powershell
.\.venv\Scripts\python.exe -B scripts/run_part3.py --run-name classical_reproduce_new --config configs/part3.yaml
```

Chọn tên lần chạy và thư mục đầu ra mới để giữ kết quả cũ.

## Fine-tune PhoBERT

Luồng PhoBERT dùng cùng nhãn và bộ chia, có bước chuẩn bị đầu vào riêng với
VnCoreNLP. Cần Java chạy được qua lệnh `java` và nhóm phụ thuộc tùy chọn:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[phobert]"
.\.venv\Scripts\python.exe scripts/setup_vncorenlp.py --output-dir tools/vncorenlp
.\.venv\Scripts\python.exe scripts/download_phobert.py --config configs/phobert.yaml
.\.venv\Scripts\python.exe -m ptit_sentiment.training.check_phobert --config configs/phobert.yaml --output outputs/phobert_environment_new.json
```

Chạy setup VnCoreNLP khi chưa có tài nguyên local; script từ chối ghi đè thư mục
đã có dữ liệu. Các script chuẩn bị tải tài nguyên mô hình/công cụ từ mạng.

```powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.training.train_phobert --splits-dir data/splits --config configs/phobert.yaml --artifact-dir artifacts/phobert_new
```

Cấu hình mặc định: `vinai/phobert-base`, max length 256, 3 epoch, learning rate
0.00002 và batch size 8. Chọn checkpoint bằng Macro-F1 validation; không đánh
giá test trong lệnh train. Artifact lưu model, tokenizer, ánh xạ nhãn, cấu hình,
metadata và thống kê truncation.

`configs/phobert_gpu4gb.yaml` cung cấp cấu hình batch nhỏ, gradient accumulation
và fp16 cho GPU 4 GB, dùng checkpoint trong cache local. Cần kiểm tra đường dẫn
cache và tài nguyên máy trước khi dùng. Cấu hình này chưa có kết quả huấn luyện
đầy đủ được xác nhận trong báo cáo hiện hành; tải pretrained hoặc chạy smoke
không được coi là đã fine-tune hoàn chỉnh.

## Đánh giá và so sánh

Chỉ đánh giá test sau khi chốt tham số/checkpoint bằng validation.
Đánh giá các pipeline đã huấn luyện ở trên:

```powershell
foreach ($name in @("bow_nb", "tfidf_nb", "bow_svm", "tfidf_svm")) {
    .\.venv\Scripts\python.exe -m ptit_sentiment.evaluation.evaluate --model "artifacts/classical_new/$name.joblib" --splits-dir data/splits --split test --output-dir "outputs/classical_new/evaluation/$name"
}
.\.venv\Scripts\python.exe -m ptit_sentiment.evaluation.compare --inputs outputs/classical_new/evaluation/bow_nb/metrics.json outputs/classical_new/evaluation/tfidf_nb/metrics.json outputs/classical_new/evaluation/bow_svm/metrics.json outputs/classical_new/evaluation/tfidf_svm/metrics.json --output outputs/classical_new/comparison.csv
```

Sau khi fine-tune PhoBERT hoàn tất, dùng cùng CLI đánh giá:

```powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.evaluation.evaluate --model artifacts/phobert_new --splits-dir data/splits --split test --output-dir outputs/phobert_new/evaluation
```

Thêm `outputs/phobert_new/evaluation/metrics.json` vào `--inputs` khi đã có kết
quả thật. Chỉ so sánh kết quả trên cùng phiên bản bộ chia/test.

Mỗi lần đánh giá xuất `metrics.json`, `classification_report.csv`,
`confusion_matrix.csv`, `confusion_matrix.png`, `predictions.csv`, `errors.csv`.
Độ đo gồm Accuracy, Precision/Recall/F1/support từng nhãn và Macro-Precision,
Macro-Recall, Macro-F1. Ma trận có hàng là nhãn thật, cột là dự đoán, thứ tự
`positive`, `neutral`, `negative`.

Kết quả truyền thống đã ghi nhận trong lần `part3_final_20261102`: TF-IDF + SVM
được chọn bằng Macro-F1 validation **0,577487**; Accuracy test **0,717949**,
Macro-F1 test **0,535355**. Phạm vi kết luận giới hạn ở bộ nhãn AI hiện hành;
chưa có căn cứ kết luận mô hình tốt nhất khi so với PhoBERT.
Xem [báo cáo hiện hành](report/report.md) và [phân tích lỗi](report/part3_error_analysis.md).

## Dự đoán một bình luận hoặc CSV

Cần artifact đã huấn luyện hoặc nhận từ gói nội bộ; artifact không kèm repo.
Ví dụ dùng pipeline vừa tạo:

```powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.predict --model artifacts/classical_new/tfidf_svm.joblib --text "Thầy giảng rất dễ hiểu."
.\.venv\Scripts\python.exe -m ptit_sentiment.predict --model artifacts/classical_new/tfidf_svm.joblib --input tests/fixtures/new_comments.csv --output outputs/predictions_new.csv
```

Với PhoBERT đã fine-tune, đổi `--model` thành `artifacts/phobert_new`; nếu di
chuyển tài nguyên VnCoreNLP, truyền thêm `--segmenter-dir tools/vncorenlp`.
CSV mới cần `id,text`, không cần `label`. Predict tải model đã lưu, không train
lại. Decision score của SVM không phải xác suất hoặc độ mạnh cảm xúc.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Các kiểm thử dùng fixture tổng hợp cho schema, gán nhãn, chia tập không rò rỉ,
đặc trưng, huấn luyện/đánh giá, lưu/tải, CLI và định dạng kết quả dùng chung.
Chạy test không thay cho huấn luyện PhoBERT đầy đủ hoặc đánh giá dữ liệu thật.

## Phân công và tài liệu

| Thành viên | Trách nhiệm |
|---|---|
| 1 | Thu thập/nhập dữ liệu, điều phối gán nhãn, kiểm tra/chia tập và ghép báo cáo |
| 2 | Tiền xử lý tiếng Việt, BoW, TF-IDF và tài liệu đặc trưng |
| 3 | NB, SVM, baseline, đánh giá và phân tích lỗi |
| 4 | Fine-tune PhoBERT, tích hợp dự đoán và hướng dẫn chạy |

Cả nhóm cùng rà nhãn và báo cáo; dùng cùng manifest, ánh xạ nhãn và bộ chia.

- [Phân công và điều kiện bàn giao](docs/task_assignment.md)
- [Hợp đồng dữ liệu](docs/data_contract.md)
- [Hướng dẫn gán nhãn](docs/labeling_guide.md)
- [Phương pháp thực nghiệm](docs/methodology.md)
- [Hướng dẫn học và luồng hệ thống](docs/learning_guide.md)
- [Báo cáo hiện hành](report/report.md)
- [Bàn giao tiền xử lý/đặc trưng](docs/nv2_handoff.md)
- [Bàn giao mô hình truyền thống](docs/part3_handoff.md)
- [Tài liệu học NB/SVM và đánh giá](docs/part3_learning.md)

## Quy tắc lưu trên GitHub

Theo dõi code, cấu hình, tests/fixtures tổng hợp và tài liệu Markdown.
`.gitignore` loại dữ liệu thật (kể cả manifest chứa ID nguồn), model, output,
archive/gói bàn giao, cache, môi trường ảo, cấu hình IDE, bằng chứng sinh tự động
trong `report/evidence/` và PDF/Word xuất báo cáo. Các file này được giữ nội bộ.
Không commit thông tin cá nhân, credentials hoặc cookies.

Dữ liệu và artifact cần được nhận qua kênh nội bộ hoặc tái tạo theo hướng dẫn
trên. Tiếp theo, nhóm cần hoàn tất fine-tune/đánh giá PhoBERT và chuẩn bị nhãn
người độc lập để xác nhận chất lượng trước khi kết luận toàn bộ thực nghiệm.
