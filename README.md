# PTIT Sentiment Analysis

Đề tài: **Xây dựng hệ thống phân loại cảm xúc bình luận của sinh viên PTIT trên các nhóm Facebook**.

Phần người 3 đã có bốn pipeline BoW/TF-IDF × NB/SVM, majority baseline và một
TF-IDF word-char + SVM bổ sung, được huấn luyện bằng CPU.
Mô hình chính là **BoW + NB**, chọn bằng Macro-F1 validation = 0,616621;
Macro-F1 test = 0,544067. [Báo cáo](report/report.md) chứa bảng kết quả đầy đủ.

Dữ liệu hiện hành: 2.600 bình luận thật mang **nhãn AI**, train/validation/test =
1.801/409/390, dataset ai_initial_4e93595ef7d1_seed42. Test đã dùng trước đó;
kết quả chưa thay đánh giá bằng nhãn người độc lập.
PhoBERT có code fine-tune, **chưa huấn luyện trên bộ chia này**.

## Cài đặt

Mở PowerShell tại thư mục gốc; khuyến nghị Python 3.12 để khớp môi trường artifact.

~~~powershell
python -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -e .
~~~

Nếu đã có .venv, chỉ cần gán $python. Phiên bản thực nghiệm lưu trong
metadata từng artifact. Gói handoff/NV3_20261021.zip có requirements được chốt phiên bản.

## Chạy mô hình đã có

Truyền bình luận gốc; pipeline tự tiền xử lý, không huấn luyện lại lúc dự đoán.

~~~powershell
& $python -m ptit_sentiment.predict --model artifacts/part3_20261012_nv2_v2/bow_nb.joblib --text "Mình chưa hài lòng với lịch học."
& $python -m ptit_sentiment.predict --model artifacts/part3_20261012_nv2_v2/bow_nb.joblib --input tests/fixtures/new_comments.csv --output outputs/predict_new.csv
~~~

CSV dự đoán cần id,text; không cần label. Nhãn: positive, neutral, negative.
SVM decision score không phải xác suất hoặc độ mạnh cảm xúc.

## Kiểm tra dữ liệu và code

~~~powershell
& $python -m ptit_sentiment.data.validate --input data/labeled/labeled_comments.csv
& $python scripts/verify_ai_dataset.py --output outputs/dataset_verification_new.json
& $python -B -m unittest discover -s tests -v
~~~

Không chia lại bộ hiện hành. Với dữ liệu mới đã được người gán xác nhận,
làm theo [quy trình dữ liệu](docs/data_contract.md) và [hướng dẫn nhãn](docs/labeling_guide.md).
Lệnh chia dưới đây chỉ minh họa bằng fixture tổng hợp, xuất riêng:

~~~powershell
& $python -m ptit_sentiment.data.split --input tests/fixtures/synthetic.csv --output-dir outputs/splits_example --config configs/classical.yaml --provenance synthetic
~~~

## Tái lập huấn luyện và đánh giá

Chọn tên lần chạy mới để bảo toàn kết quả hiện có. Runner kiểm tra dữ liệu/NV2,
train trên train, chọn bằng validation, khóa artifact rồi đánh giá test:

~~~powershell
& $python scripts/run_part3.py --run-name part3_reproduce_new --config configs/part3_v2.yaml --extended-config configs/part3_v2_extended.yaml
~~~

Có thể chạy từng bước cho bốn pipeline bắt buộc và baseline:

~~~powershell
& $python -m ptit_sentiment.training.train_classical --splits-dir data/splits --config configs/part3_v2.yaml --artifact-dir artifacts/manual_new --output-dir outputs/manual_new/training
& $python -m ptit_sentiment.evaluation.evaluate --model artifacts/manual_new/bow_nb.joblib --splits-dir data/splits --output-dir outputs/manual_new/evaluation/bow_nb --split test
& $python -m ptit_sentiment.evaluation.evaluate --model artifacts/manual_new/tfidf_svm.joblib --splits-dir data/splits --output-dir outputs/manual_new/evaluation/tfidf_svm --split test
& $python -m ptit_sentiment.evaluation.compare --inputs outputs/manual_new/evaluation/bow_nb/metrics.json outputs/manual_new/evaluation/tfidf_svm/metrics.json --output outputs/manual_new/comparison.csv
~~~

Chỉ đánh giá test sau khi đã chốt lựa chọn. Không tune theo test.
Để xuất bản bằng chứng mới từ lần chạy hiện hành, chọn thư mục chưa tồn tại:

~~~powershell
& $python scripts/export_report_evidence.py --run-dir outputs/part3_20261012_nv2_v2 --splits-dir data/splits --output-dir outputs/report_evidence_new
~~~

## PhoBERT — phần người 4

Cài thêm nhóm phụ thuộc, Java và VnCoreNLP; checkpoint cần tải một lần.
Lệnh dưới đây là hướng dẫn thực nghiệm tiếp theo, chưa phải kết quả đã hoàn thành.

~~~powershell
& $python -m pip install -e ".[phobert]"
& $python scripts/setup_vncorenlp.py --output-dir tools/vncorenlp
& $python scripts/download_phobert.py --config configs/phobert.yaml
& $python -m ptit_sentiment.training.train_phobert --splits-dir data/splits --config configs/phobert_gpu4gb.yaml --artifact-dir artifacts/phobert_current_new
& $python -m ptit_sentiment.evaluation.evaluate --model artifacts/phobert_current_new --splits-dir data/splits --output-dir outputs/phobert_current_new/evaluation --split test
~~~

Dùng cùng manifest/ánh xạ nhãn, chọn checkpoint bằng Macro-F1 validation.
Đầu vào PhoBERT dùng RDRSegmenter riêng; xem [phương pháp](docs/methodology.md).
Không dùng pretrained checkpoint hoặc điểm smoke làm kết quả fine-tune.

## Những thư mục cần dùng

| Đường dẫn | Nội dung |
|---|---|
| src/ptit_sentiment/, features.py | Code dữ liệu, tiền xử lý, đặc trưng, train, evaluate và predict |
| configs/ | Cấu hình dùng được; thực nghiệm người 3 dùng part3_v2 và part3_v2_extended |
| data/ | Một CSV labeled, ba CSV bộ chia và manifest |
| tests/ | Kiểm thử và fixture tổng hợp, tách khỏi thực nghiệm |
| artifacts/part3_20261012_nv2_v2/ | Pipeline hiện hành và metadata |
| outputs/part3_20261012_nv2_v2/ | Log train, khóa lựa chọn, audit và kết quả test |
| outputs/part3_completion_20261021/ | Kết quả validation, bảng tổng hợp và lựa chọn chính |
| outputs/test_human_review_20261021/ | Excel có nhãn nháp bạn đã nhập, được giữ nguyên |
| outputs/test_ai_labeled_20261021/ | Excel 390 nhãn AI và bản nháp người dùng để đối chiếu |
| report/report.md, report/evidence/ | Báo cáo chính và bằng chứng đúng bộ chia hiện hành |
| handoff/ | Gói bàn giao mô hình và dữ liệu tối giản |
| outputs/archives/ | Nguồn/đối soát dữ liệu và bản sao lưu file đã dọn |

Tài liệu: [phân công](docs/task_assignment.md), [bàn giao NV2](docs/nv2_handoff.md),
[bàn giao người 3/API người 4](docs/part3_handoff.md),
[học toàn dự án](docs/learning_guide.md), [học NB/SVM](docs/part3_learning.md),
[phân tích lỗi](report/part3_error_analysis.md).

Đợt dọn 23/10/2026 giữ một bản code/tài liệu/kết quả hiện hành.
Script dò/debug, bản demo/smoke, checkpoint bộ chia cũ và tài liệu lặp được lưu vào
outputs/archives/project_cleanup_20261023.zip, kèm manifest SHA-256 và hướng dẫn khôi phục.
Nhật ký kiểm tra ở outputs/cleanup_20261023/.
Môi trường .venv, cache tải mô hình và tài nguyên VnCoreNLP vẫn được giữ để chạy dự án.
