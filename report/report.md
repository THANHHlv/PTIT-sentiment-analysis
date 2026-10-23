# Phần người 3 — NB, SVM, đánh giá và phân tích lỗi

Bản bàn giao hoàn thiện và kiểm chứng lại ngày 21/10/2026, sử dụng mô hình đã khóa của lần 12/10.
Không huấn luyện/chọn lại từ test. Lệnh mặc định hiện dùng configs/part3_v2.yaml.
67 tests đạt; đã bổ sung kiểm tra cấu hình TF-IDF, điểm raw-text validation và gói bàn giao.
Kết quả validation đầy đủ ở outputs/part3_completion_20261021/validation/.
Bảng tổng hợp và lựa chọn chính: results.csv, primary_model.json trong thư mục đó.
Gói riêng cho nhóm: handoff/NV3_20261021.zip; nguồn nhãn AI/test đã dùng được ghi trong metadata.
Xem [bàn giao hiện hành](../docs/part3_handoff.md).


Đã hoàn thành sửa tích hợp, audit dữ liệu, train và evaluate bốn pipeline bắt buộc,
majority baseline và một TF-IDF word-char + SVM bổ sung bằng CPU.
Lần chạy: **part3_20261012_nv2_v2**; artifact và output hiện hành được giữ nguyên.
Toàn bộ nhãn là AI ban đầu; test đã dùng ngày 09/10. Đây là so sánh bổ sung
trên cùng test, **chưa phải đánh giá độc lập trên dữ liệu mới hoặc nhãn người**.
PhoBERT chưa được fine-tune/đánh giá trong nhiệm vụ người 3.


Ngày 21/10, người dùng chọn tiếp tục sử dụng bộ nhãn AI ban đầu. Bản Excel 390 câu
được điền lại đúng nhãn đang dùng trong thực nghiệm, không tạo test label từ dự đoán NB/SVM.
25 nhãn người dùng nhập được bảo toàn như bản nháp, chưa được dùng làm human_confirmed.
Không có thay đổi nhãn, bộ chia hoặc huấn luyện lại; các kết quả bên dưới giữ nguyên.

## 1. Dữ liệu và bàn giao NV2

Đề tài: **Xây dựng hệ thống phân loại cảm xúc bình luận của sinh viên PTIT trên các nhóm Facebook**.
Dataset `ai_initial_4e93595ef7d1_seed42`, seed 42. Giữ bộ chia theo nhóm bài và các
ràng buộc trùng/gần trùng hiện có; không thu thập, đổi nhãn hoặc chia lại.

| Tập | Số câu | Số bài | positive | neutral | negative |
|---|---:|---:|---:|---:|---:|
| Train | 1801 | 326 | 254 | 1246 | 301 |
| Validation | 409 | 63 | 58 | 283 | 68 |
| Test | 390 | 57 | 50 | 271 | 69 |
| Tổng | 2600 | 446 | 362 | 1800 | 438 |

2600/2600 dòng có `label_source=ai`, `annotation_status=ai_labeled`;
`human_labels_confirmed=false`. Không giao ID/post_id/nội dung chuẩn hóa hoặc các
cặp gần trùng đã ghi trong manifest. Tổng ba tập khớp mọi cột của bảng labeled.
Nhãn chung positive=0, neutral=1, negative=2.

Sau pull, hai module mô hình/train có marker merge và bốn CSV sai hash do CRLF/LF.
Đã ghép đúng code NV2/NV3 và khôi phục byte từ ZIP sau khi kiểm tra bản chuẩn
khớp manifest và bản hiện tại chỉ khác xuống dòng. Manifest/nhãn/bộ chia giữ nguyên.
Bản cũ được lưu trong `outputs/archives/project_cleanup_20261023.zip`; `.gitattributes` bảo vệ EOL.
[Audit NV2 mới](../docs/nv2_handoff.md) ghi chi tiết và các tài liệu bàn giao còn thiếu.

Manifest SHA-256: `c743878f73ef20ef786f46f46c62511b894a10891c2893de8d75cb753522e4e3`.
Test SHA-256: `afcb8ce871c14ea258fc02a8ec56596a2c7e2e034b88d2437f2cdffdc0184312`.

## 2. Tiền xử lý, đặc trưng và mô hình

Text gốc → NFC/khoảng trắng/ký tự lỗi → che email/điện thoại/@handle, lowercase,
thay URL, mở rộng ko/khum/hok → Underthesea → vectorizer → classifier.
Giữ k/dc/đc/mn/sv do đa nghĩa; giữ phủ định/emoji/dấu câu, không loại stop words.
Pipeline lưu nhận text gốc; không train lại khi predict.

BoW là số đếm; TF-IDF học vocabulary/DF/IDF chỉ từ train và chuẩn hóa L2.
Word unigram có 3569 cột, unigram–bigram có 17841 cột;
min_df=1, max_df=1.0, max_features=20000. Ma trận sparse không âm.
Các cấu hình từ có 0/9/4 vector 0 ở train/validation/test.

MultinomialNB học prior lớp và phân bố token có làm trơn alpha, giả định độc lập có điều kiện.
LinearSVC học điểm tuyến tính one-vs-rest; C điều chỉnh phạt vi phạm lề, không xuất xác suất.
Baseline học lớp phổ biến nhất chỉ từ train: neutral.
SVM dùng L2, squared hinge, dual=auto, tol=1e-4, max_iter=10000, random_state=42;
bắt ConvergenceWarning, tăng max_iter=100000 và thử lại một lần khi cần.

Nhánh bổ sung ghép TF-IDF word và char_wb n-gram (3,5), trần 20000 cột mỗi nhánh.
Char lấy từ cùng text đã xử lý/tách từ, không phải text gốc. Hai nhánh chuẩn hóa L2
riêng rồi nối, không chuẩn hóa lại toàn vector. Artifact được chọn có 3569+20000=23569 cột,
0 vector 0 ở train/validation. Audit cả vocabulary/IDF của hai nhánh với train;
heldout không đổi tham số. Công thức NB/SVM/độ đo ở [tài liệu học](../docs/part3_learning.md).

## 3. Thiết kế thực nghiệm và khóa lựa chọn

[part3_v2.yaml](../configs/part3_v2.yaml): 6 ứng viên mỗi tổ hợp,
ngram từ (1,1)/(1,2), alpha NB=0.1/0.5/1, C SVM=0.1/1/10;
emoji keep, SVM không cân bằng lớp, TF-IDF word. Tổng **24 lượt fit** bắt buộc.

[part3_v2_extended.yaml](../configs/part3_v2_extended.yaml): riêng TF-IDF + SVM,
2 ngrams × 3 C × 2 emoji (keep/keep_and_map) × 2 class weights (None/balanced)
× 2 biểu diễn (word/word_char) = **48 lượt fit bổ sung**.
Không thử trigram/min_df/sublinear_tf trong lần này; dùng tên `tfidf_svm_extended` riêng.

Mỗi ứng viên chỉ fit train, chọn bằng Macro-F1 validation trên đủ ba nhãn,
zero_division=0. Hòa lấy ứng viên đầu trong YAML; không refit train+validation.
Cache tiền xử lý riêng theo emoji; vectorizer/classifier từng ứng viên độc lập.
Tổng 72 ứng viên; **60 lượt SVM hội tụ ngay**, không retry/cảnh báo hội tụ.
Raw-text validation score của pipeline đã lưu khớp điểm lựa chọn.

| Mô hình | Word n-gram | alpha / C | Macro-F1 validation |
|---|---|---|---:|
| BoW + NB | unigram | alpha=0.1 | 0.616621 |
| TF-IDF word + NB | unigram | alpha=0.1 | 0.504317 |
| BoW + SVM | unigram | C=1 | 0.558472 |
| TF-IDF word + SVM | unigram | C=1 | 0.577487 |
| TF-IDF word-char + SVM (bổ sung) | unigram | C=1; balanced | 0.605381 |

BoW+NB đứng đầu validation nên được chọn làm mô hình chính để phân tích lỗi.
Nhánh bổ sung chọn word-char, unigram, C=1, balanced, emoji keep.

Bảng sau chọn n-gram/C tốt nhất **trong từng nhóm bằng validation**;
vì C có thể khác nhau, chênh lệch không phải thí nghiệm chỉ đổi một yếu tố.
Chi tiết ở `training_extended/` và `validation_ablation.csv`.

| Biểu diễn | Emoji | Class weight | Word n-gram | C | Macro-F1 validation |
|---|---|---|---|---:|---:|
| word | keep | None | unigram | 1 | 0.577487 |
| word | keep | balanced | unigram | 1 | 0.576425 |
| word-char | keep | None | unigram | 10 | 0.579014 |
| word-char | keep | balanced | unigram | 1 | 0.605381 |
| word | keep_and_map | None | unigram | 1 | 0.577487 |
| word | keep_and_map | balanced | unigram | 1 | 0.576425 |
| word-char | keep_and_map | None | unigram | 10 | 0.579014 |
| word-char | keep_and_map | balanced | unigram | 1 | 0.605381 |

Unicode emoji được nhận diện ở 3 train/10 validation/0 test, nhưng **0 câu đổi đầu ra**
giữa keep và keep_and_map vì không có emoji thuộc bảng ánh xạ hiện tại.
Điểm bằng nhau chưa đánh giá được hiệu quả ánh xạ trên dữ liệu có độ phủ phù hợp;
48 lượt fit chỉ có 24 tổ hợp tạo đầu vào khác nhau. Xem `emoji_coverage.json`.

`all_selection_lock.json` khóa sáu artifact lúc **13:28:48 UTC ngày 12/10/2026**,
trước mọi lệnh test của lần mới; `test_predictions_computed=false` khi khóa.
Không thay lựa chọn sau đọc test; hash artifact/lock được kiểm tra cuối lần chạy.

## 4. Kết quả trên test nhãn AI đã dùng trước đó

Cùng 390 mẫu, chưa có test mới/nhãn người độc lập. Macro theo đúng ba nhãn,
undefined metric=0. Confusion matrix hàng nhãn AI/cột dự đoán,
thứ tự positive/neutral/negative. Test không chọn tham số trong lần mới.

| Mô hình | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Số sai |
|---|---:|---:|---:|---:|---:|
| BoW + NB | 0.694872 | 0.555395 | 0.535619 | 0.544067 | 119 |
| TF-IDF word + NB | 0.723077 | 0.610838 | 0.446135 | 0.466176 | 108 |
| BoW + SVM | 0.710256 | 0.561939 | 0.537563 | 0.547529 | 113 |
| TF-IDF word + SVM | 0.717949 | 0.571906 | 0.515976 | 0.535355 | 110 |
| Majority baseline | 0.694872 | 0.231624 | 0.333333 | 0.273323 | 119 |
| TF-IDF word-char + SVM (bổ sung) | 0.707692 | 0.570426 | 0.558009 | 0.563401 | 114 |

![Cùng 390 mẫu test nhãn AI](../outputs/part3_20261012_nv2_v2/comparison.png)

| Mô hình | Nhãn | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| BoW + NB | positive | 0.4390 | 0.3600 | 0.3956 | 50 |
| BoW + NB | neutral | 0.7943 | 0.8266 | 0.8101 | 271 |
| BoW + NB | negative | 0.4328 | 0.4203 | 0.4265 | 69 |
| TF-IDF word + NB | positive | 0.5385 | 0.1400 | 0.2222 | 50 |
| TF-IDF word + NB | neutral | 0.7457 | 0.9520 | 0.8363 | 271 |
| TF-IDF word + NB | negative | 0.5484 | 0.2464 | 0.3400 | 69 |
| BoW + SVM | positive | 0.4359 | 0.3400 | 0.3820 | 50 |
| BoW + SVM | neutral | 0.8105 | 0.8524 | 0.8309 | 271 |
| BoW + SVM | negative | 0.4394 | 0.4203 | 0.4296 | 69 |
| TF-IDF word + SVM | positive | 0.4545 | 0.3000 | 0.3614 | 50 |
| TF-IDF word + SVM | neutral | 0.7895 | 0.8856 | 0.8348 | 271 |
| TF-IDF word + SVM | negative | 0.4717 | 0.3623 | 0.4098 | 69 |
| Majority baseline | positive | 0.0000 | 0.0000 | 0.0000 | 50 |
| Majority baseline | neutral | 0.6949 | 1.0000 | 0.8200 | 271 |
| Majority baseline | negative | 0.0000 | 0.0000 | 0.0000 | 69 |
| TF-IDF word-char + SVM | positive | 0.4419 | 0.3800 | 0.4086 | 50 |
| TF-IDF word-char + SVM | neutral | 0.8123 | 0.8303 | 0.8212 | 271 |
| TF-IDF word-char + SVM | negative | 0.4571 | 0.4638 | 0.4604 | 69 |

Trong bốn tổ hợp bắt buộc, BoW+SVM có Macro-F1 test cao nhất (0,547529), nhưng
BoW+NB vẫn là lựa chọn theo validation. Nhánh bổ sung đứng đầu bảng sáu mô hình
về Macro-F1 (0,563401), hơn TF-IDF word+SVM khoảng 0,0280;
Accuracy thấp hơn (0,707692 so với 0,717949), recall hai lớp thiểu số cao hơn.
TF-IDF+NB có Accuracy cao nhất (0,723077), song recall positive chỉ 0,14.
Baseline Accuracy=0,694872 nhưng F1 positive/negative=0.
Chưa kiểm định ý nghĩa thống kê; không kết luận thuật toán tốt nhất nói chung.

Confusion matrix của mô hình bổ sung:

| Nhãn AI / dự đoán | positive | neutral | negative |
|---|---:|---:|---:|
| positive | 19 | 22 | 9 |
| neutral | 17 | 225 | 29 |
| negative | 7 | 30 | 32 |

![Confusion matrix của nhánh bổ sung](../outputs/part3_20261012_nv2_v2/evaluation/tfidf_svm_extended/confusion_matrix.png)

## 5. Phân tích lỗi

[25 trích đoạn sai đã ẩn danh](part3_error_analysis.md) bao phủ sáu hướng nhầm
của BoW+NB; từng trích đoạn đã đối chiếu `errors.csv`.
Có câu ngắn, viết tắt, phủ định gián tiếp, mỉa mai và trộn sắc thái;
tách quan sát/giả thuyết, không sửa nhãn AI từ dự đoán.
BoW+NB nhầm 24 positive và 34 negative thành neutral (58/119 lỗi).

`error_groups.csv`/`errors_by_post.csv` mô tả nhóm lỗi/bài; nhóm tự động có giao nhau,
không phải taxonomy nguyên nhân. Nhóm phủ định có 69 mẫu, BoW+NB sai 18,
nhánh bổ sung sai 14; đây là mô tả test, không dùng để điều chỉnh tiếp.
Mẫu định tính có chủ đích, không dùng 25 câu suy ra tỷ lệ nguyên nhân toàn dữ liệu.

## 6. Kiểm chứng và tái lập

**67/67 tests đạt sau đợt hoàn thiện**; audit dữ liệu/NV2 và năm pipeline đã chọn đạt.
Sáu mô hình tải lại giữ đúng 390 dự đoán và thứ tự ID/text/nhãn;
metrics tính lại khớp, số errors khớp; CLI câu/CSV/câu rỗng/artifact thiếu đạt.
CSV/manifest và hash lock giữ nguyên sau train/evaluate. Đã xem biểu đồ và confusion matrix.
Phiên bản môi trường lưu trong metadata; chạy CPU, không huấn luyện PhoBERT.

Lần chạy bị ngắt sau train rồi tiếp tục từ bước thành công, không train lại.
`run_context.json` giữ snapshot ban đầu; `resume_context` ghi lúc tiếp tục;
`commands.json` có lệnh/exit code/stdout/stderr/thời gian của cả hai đoạn.
315,56 giây là tổng thời gian CLI đã ghi, không phải toàn phiên gồm khoảng gián đoạn.
200,69 giây trong `verification.json` là đoạn runner tiếp tục, không phải tổng lần chạy.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python scripts/run_part3.py --run-name part3_v2_rerun_01 --config configs/part3_v2.yaml --extended-config configs/part3_v2_extended.yaml
& $python -m ptit_sentiment.predict --model artifacts/part3_20261012_nv2_v2/bow_nb.joblib --text "Mình chưa hài lòng với lịch học."
```

Chọn run-name/output mới. Artifact ở `artifacts/part3_20261012_nv2_v2/`, nhánh bổ sung
ở `extended/`. Output ở `outputs/part3_20261012_nv2_v2/`: metrics.json,
classification_report.csv, confusion_matrix.png, predictions.csv, errors.csv,
error_groups.csv, errors_by_post.csv, comparison.csv/.png. Raw output giữ local.
API/lệnh cho người 4: [bàn giao mới](../docs/part3_handoff.md).

## 7. Giới hạn và việc tiếp theo

Nhãn AI, lệch neutral, một bộ chia; phát hiện gần trùng chỉ bảo vệ ràng buộc đã nhận diện.
Test đã dùng, nhiều ứng viên chọn trên cùng validation có nguy cơ khớp riêng validation.
Chưa có nhãn người độc lập/kiểm định thống kê; emoji map chưa có độ phủ phù hợp.
Regex chưa che hết tên người nên ví dụ báo cáo được rà riêng.

Phần người 3 đã có code, train/evaluate thật và tài liệu. Chưa tuyên bố toàn đề tài
hoàn tất vì PhoBERT thuộc người 4 và nhãn người độc lập chưa có.
Tiếp theo: người 4 fine-tune PhoBERT cùng bộ chia, khóa checkpoint bằng validation,
dùng API đánh giá/metadata chung; nhóm chuẩn bị test mới mang nhãn người để xác nhận độc lập.
Lý thuyết/tài liệu tham khảo phiên bản đã dùng ở [tài liệu học](../docs/part3_learning.md)
và [hướng dẫn phương pháp](../docs/methodology.md).
