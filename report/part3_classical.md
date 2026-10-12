# Thực nghiệm phần 3: Naive Bayes, SVM, đánh giá và phân tích lỗi

Cập nhật 09/10/2026. Lần chạy **`part3_20261009_cpu`** hoàn thành bằng CPU;
code chạy được, đã train và đánh giá bốn tổ hợp truyền thống cùng baseline trên bình luận thật.
**Toàn bộ nhãn là AI ban đầu**, không có nhãn người xác nhận độc lập.
Phần này không thực hiện PhoBERT và không tuyên bố hoàn thành toàn bộ đề tài.

## 1. Dữ liệu và kiểm tra bàn giao

Tên đề tài: **Xây dựng hệ thống phân loại cảm xúc bình luận của sinh viên PTIT trên các nhóm Facebook**.
Đầu vào là bình luận tiếng Việt; đầu ra đúng một nhãn positive, neutral hoặc negative.
Nguồn dữ liệu lưu trong hồ sơ hiện có là hai nhóm PTIT `2k5ptit` và `584397217391365`.
Không truy cập lại Facebook hoặc thu thập thêm trong lần này.

Đã tái sử dụng `TextPreprocessor`, `make_vectorizer` và `features.py` của NV2;
audit thực tế tại [part2_review.md](../docs/part2_review.md).
Các file ví dụ/kế hoạch/báo cáo riêng được nêu trong bàn giao NV2 nhưng không có ở workspace đã được ghi rõ.
Cảnh báo không khớp hash trong bàn giao không tái hiện: bốn CSV khớp manifest và bản ZIP lưu trữ.
Không thay hash, nhãn, text hay cách chia để tiếp tục chạy.

| Tập | Số câu | Số bài | positive | neutral | negative | Tỷ lệ câu |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1801 | 326 | 254 | 1246 | 301 | 69,27% |
| Validation | 409 | 63 | 58 | 283 | 68 | 15,73% |
| Test | 390 | 57 | 50 | 271 | 69 | 15,00% |
| Tổng | 2600 | 446 | 362 | 1800 | 438 | 100% |

`label_source=ai`, `annotation_status=ai_labeled` ở 2600/2600 dòng;
`human_labels_confirmed=false`. Không đổi AI thành human_confirmed.
Validator kiểm tra cột, chuỗi ID, ô rỗng, nhãn sai, trùng ID/text chuẩn hóa.
`load_split_set` kiểm tra hash, đúng thành viên/thứ tự ID, không giao ID/post_id/nội dung,
ràng buộc nhóm và gần trùng đã nhận diện. Tổng ba tập khớp toàn bộ cột của bảng labeled.
Bộ chia theo post_id và các thành phần trùng/gần trùng, seed 42; giữ nguyên cho người 4.

Dataset: `ai_initial_4e93595ef7d1_seed42`.

```text
Manifest SHA-256:
c743878f73ef20ef786f46f46c62511b894a10891c2893de8d75cb753522e4e3
Labeled SHA-256:
4e93595ef7d19e1b60d71e44f83cdeabc2790ef15b0113831b0e59ee2a331dc4
Test SHA-256:
afcb8ce871c14ea258fc02a8ec56596a2c7e2e034b88d2437f2cdffdc0184312
```

## 2. Tiền xử lý, đặc trưng và mô hình

Pipeline đầy đủ: text gốc → NFC/khoảng trắng/ký tự lỗi → che email/điện thoại/@handle,
lowercase, thay URL, slang → Underthesea → BoW/TF-IDF → classifier.
Giữ phủ định, emoji, dấu câu; stop words rỗng. Không ghi đè text gốc.
Câu thiếu/rỗng hoặc rỗng sau xử lý báo lỗi; câu hợp lệ ngoài vocabulary được giữ thành vector 0.
Audit có 9 vector 0 ở validation, 4 ở test và 0 ở train.

BoW đếm token; TF-IDF dùng số đếm nhân IDF từ train, chuẩn hóa L2.
Unigram có 3543 cột; unigram–bigram có 17672 cột. Tất cả ma trận sparse, không âm;
không chuyển toàn bộ thành dense. `min_df=1`, `max_df=1.0`, `max_features=20000` giữ cấu hình NV2
vì dữ liệu nhỏ và từ hiếm có thể mang cảm xúc; không điều chỉnh các giá trị này theo test.

MultinomialNB học prior lớp và trọng số token theo lớp. Dự đoán lấy lớp tối đa
`log P(c) + Σ_j x_j log θ_cj`, với `θ_cj=(N_cj+alpha)/(Σ_k N_ck+alpha×V)`.
Alpha làm trơn tránh xác suất bằng 0; tăng alpha làm phân bố bớt cực đoan.
Giả định độc lập có điều kiện không mô tả đầy đủ tương tác từ. TF-IDF là đầu vào thực không âm
dùng thực dụng với NB, thay cho số đếm nguyên trong giả định ban đầu.

LinearSVC học trọng số và intercept cho ba lớp theo one-vs-rest; lấy lớp có điểm tuyến tính lớn nhất.
C cân bằng phạt vi phạm lề và regularization; C lớn không đảm bảo khái quát hóa tốt hơn.
Lần này dùng L2, squared hinge, dual=auto, tol=1e-4, max_iter=10000, random_state=42.
SVM không xuất xác suất. Baseline chỉ học lớp phổ biến nhất ở train: neutral.

## 3. Thiết kế thực nghiệm và khóa lựa chọn

Cấu hình tại [configs/part3.yaml](../configs/part3.yaml):
alpha NB `{0.1,0.5,1.0}`, C SVM `{0.1,1.0,10.0}`,
ngram_range `{(1,1),(1,2)}`. Mỗi tổ hợp có 6 ứng viên, **24 ứng viên tổng cộng**.
Mỗi vectorizer/classifier chỉ fit train; validation chỉ transform và predict.
Tách từ bằng bộ xử lý không học tham số một lần cho mỗi tập train/validation để tiết kiệm CPU;
không xử lý hai lần khi dự đoán từ pipeline lưu.

Chọn bằng Macro-F1 validation, tính trên đủ ba nhãn, zero_division=0.
Nếu bằng điểm, lấy ứng viên đầu trong YAML: unigram trước, alpha/C nhỏ trước.
Không dùng GridSearchCV hoặc cross-validation; không refit train+validation.
Lưu mọi điểm/thời gian/cấu hình ứng viên trong `training/*_validation_search.json`.
SVM bắt cảnh báo hội tụ và có cơ chế tăng max_iter/thử lại; **12 ứng viên SVM đều hội tụ ngay**,
không cần retry. Số vòng lặp nằm trong log từng ứng viên.

| Mô hình | ngram_range | alpha / C | Macro-F1 validation | Thời gian tìm 6 cấu hình (s) |
|---|---|---|---:|---:|
| BoW + NB | (1,1) | alpha=0.1 | 0.618035 | 0.265 |
| TF-IDF + NB | (1,1) | alpha=0.1 | 0.502193 | 0.322 |
| BoW + LinearSVC | (1,1) | C=1.0 | 0.562747 | 0.531 |
| TF-IDF + LinearSVC | (1,1) | C=1.0 | 0.571935 | 0.467 |

Thời gian tìm kiếm chưa cộng phần tách từ dùng chung 7,056 s và kiểm tra serialize.
Toàn lệnh train mất 11,247 s. `selection_lock.json` được ghi lúc
**11:15:15 ngày 09/10/2026 (Asia/Saigon)** trước lệnh đánh giá test đầu tiên;
khóa có hash của cả năm artifact và ghi `test_predictions_computed=false` tại thời điểm khóa.
BoW+NB đứng đầu validation nên được dùng cho phân tích lỗi định tính.

## 4. Kết quả test thực tế

Các mô hình đã khóa được đánh giá trên cùng 390 mẫu test, không train thêm.
Accuracy là tỷ lệ đúng; Precision/Recall/F1 tính riêng từng lớp;
macro là trung bình đều của đúng ba lớp, không theo support.
Undefined metric bằng 0. Confusion Matrix giữ thứ tự positive, neutral, negative;
**hàng là nhãn thật AI, cột là nhãn dự đoán**.

| Mô hình | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Số sai |
|---|---:|---:|---:|---:|---:|
| BoW + NB | 0.694872 | 0.553874 | 0.533784 | **0.541757** | 119 |
| TF-IDF + NB | **0.717949** | 0.599122 | 0.440075 | 0.458428 | 110 |
| BoW + LinearSVC | 0.694872 | 0.544462 | 0.522981 | 0.531932 | 119 |
| TF-IDF + LinearSVC | 0.707692 | 0.558728 | 0.507455 | 0.525456 | 114 |
| Majority baseline | 0.694872 | 0.231624 | 0.333333 | 0.273323 | 119 |

![Accuracy và Macro-F1 trên cùng test nhãn AI](../outputs/part3_20261009_cpu/comparison.png)

| Mô hình | Nhãn | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| BoW + NB | positive | 0.4359 | 0.3400 | 0.3820 | 50 |
| BoW + NB | neutral | 0.7972 | 0.8266 | 0.8116 | 271 |
| BoW + NB | negative | 0.4286 | 0.4348 | 0.4317 | 69 |
| TF-IDF + NB | positive | 0.5385 | 0.1400 | 0.2222 | 50 |
| TF-IDF + NB | neutral | 0.7428 | 0.9483 | 0.8331 | 271 |
| TF-IDF + NB | negative | 0.5161 | 0.2319 | 0.3200 | 69 |
| BoW + LinearSVC | positive | 0.4250 | 0.3400 | 0.3778 | 50 |
| BoW + LinearSVC | neutral | 0.7993 | 0.8376 | 0.8180 | 271 |
| BoW + LinearSVC | negative | 0.4091 | 0.3913 | 0.4000 | 69 |
| TF-IDF + LinearSVC | positive | 0.4412 | 0.3000 | 0.3571 | 50 |
| TF-IDF + LinearSVC | neutral | 0.7822 | 0.8745 | 0.8258 | 271 |
| TF-IDF + LinearSVC | negative | 0.4528 | 0.3478 | 0.3934 | 69 |
| Majority baseline | positive | 0.0000 | 0.0000 | 0.0000 | 50 |
| Majority baseline | neutral | 0.6949 | 1.0000 | 0.8200 | 271 |
| Majority baseline | negative | 0.0000 | 0.0000 | 0.0000 | 69 |

Confusion Matrix của BoW+NB:

| Nhãn thật AI \ dự đoán | positive | neutral | negative |
|---|---:|---:|---:|
| positive | 17 | 24 | 9 |
| neutral | 16 | 224 | 31 |
| negative | 6 | 33 | 30 |

Ví dụ 33 negative bị dự đoán neutral; đường chéo có 271 câu đúng.
Ma trận và báo cáo từng mô hình ở `evaluation/<model>/confusion_matrix.csv/.png`,
`classification_report.csv` và `metrics.json`.

## 5. So sánh và phân tích lỗi

BoW+NB có Macro-F1 cao nhất **trong năm mô hình và bộ test này**, cao hơn baseline khoảng
0,2684 điểm. Accuracy của BoW+NB bằng baseline nhưng đã nhận ra một phần positive/negative;
baseline có F1=0 ở cả hai lớp đó. Điều này cho thấy cần đọc macro và per-label thay vì chỉ Accuracy.
TF-IDF+NB có Accuracy cao nhất nhưng recall positive=0,14 và negative≈0,2319,
do đó Macro-F1 thấp hơn các tổ hợp còn lại.

Với NB, BoW hơn TF-IDF khoảng 0,0833 Macro-F1; với SVM, BoW hơn TF-IDF khoảng 0,0065.
Với BoW, NB hơn SVM khoảng 0,0098; với TF-IDF, SVM hơn NB khoảng 0,0670.
Chưa kiểm định ý nghĩa thống kê, không kết luận ưu thế thuật toán phổ quát.
Unigram thắng unigram–bigram ở validation của cả bốn tổ hợp;
giả thuyết là bigram thưa trên dữ liệu nhỏ, nhưng chưa có thí nghiệm riêng chứng minh nguyên nhân.
Không lấy điểm test của các cấu hình thua validation để điều chỉnh lựa chọn.

[Phân tích 25 câu sai](part3_error_analysis.md) bao phủ sáu hướng nhầm lớp,
tách rõ quan sát và giả thuyết. Các ví dụ đã bỏ tên người và ID Facebook.
Quan sát có câu hỏi tu từ/phủ định kép, mỉa mai `vote 1 sao, xin cảm ơn`,
tiếng lóng kéo dài, tên chương trình `chất lượng cao`, câu quá ngắn và câu trộn sắc thái.
Không tự sửa nhãn từ những ví dụ sai; nhãn AI cũng cần người kiểm tra độc lập.

## 6. Kiểm chứng, truy vết và tái lập

Lần chạy bắt đầu **11:14:28** và kết thúc **11:16:25 ngày 09/10/2026 (Asia/Saigon)**,
tổng 116,936 s gồm audit, tests, train, evaluate, compare và verification.
Thiết bị CPU. Python 3.12.9, scikit-learn 1.5.2, pandas 2.2.3, numpy 1.26.4,
Underthesea 9.5.0, joblib 1.4.2, matplotlib 3.9.2, PyYAML 6.0.2.
Không import/chạy thư viện PhoBERT để huấn luyện phần này.

57/57 tests đạt; audit dữ liệu thật và ma trận NV2 đạt. Đã kiểm tra:
thứ tự ID/text/nhãn, không đổi vocabulary/IDF từ heldout, ma trận sparse/không âm,
24 lượt fit/validation, hội tụ SVM, serialize/reload toàn bộ 390 dự đoán của từng artifact,
tính lại metrics từ predictions, đối chiếu số errors, predict câu/CSV và lỗi câu rỗng/artifact thiếu.
Đã xem trực quan biểu đồ so sánh và confusion matrix. Tất cả CSV/manifest giữ nguyên hash cuối lần chạy.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python scripts/run_part3.py --run-name part3_rerun_01
& $python -m ptit_sentiment.predict --model artifacts/part3_20261009_cpu/bow_nb.joblib --text "Mình chưa hài lòng với lịch học."
```

Chọn run-name mới; lệnh không ghi đè lần chạy cũ. Chi tiết lệnh train/evaluate/compare và
API chung cho người 4 ở [part3_handoff.md](../docs/part3_handoff.md).
`commands.json` ghi lệnh/exit code/stdout/stderr/thời gian;
`run_context.json` ghi commit/trạng thái Git/hash code;
`training/selection_lock.json` ghi lựa chọn trước test; `verification.json` ghi kiểm chứng.
Mỗi model có pipeline joblib và metadata đi kèm. Kết quả từng model có metrics.json,
classification_report.csv, confusion_matrix.png, predictions.csv, errors.csv và các CSV bổ sung.
Predict không fit lại; không coi decision score SVM là xác suất.

## 7. Hạn chế và phần tiếp theo

Nhãn AI chỉ cho phép đo độ phù hợp với bộ AI; **chưa phải đánh giá chất lượng trên nhãn người độc lập**.
Neutral chiếm 69,23%; hai lớp còn lại ít mẫu. Chia theo bài giúp chống rò rỉ nhưng chỉ là
một lần chia, chưa đánh giá dao động qua nhiều seed hay khả năng khái quát ngoài hai nhóm.
Ràng buộc gần trùng chỉ bảo vệ các mẫu được phương pháp hiện tại nhận diện, không đảm bảo mọi diễn đạt gần nghĩa.
Câu rời hội thoại thiếu ngữ cảnh, slang/emoji/tên riêng và nhãn mơ hồ còn gây lỗi.
Che thông tin nhạy cảm bằng regex chưa đảm bảo ẩn mọi tên người; output chi tiết giữ local.

Không có phần 3 bị chặn. Bước tiếp theo là người 4 fine-tune PhoBERT trên cùng bộ chia,
chọn checkpoint trên validation rồi dùng bộ xuất đánh giá chung. Nhóm cần rà nhãn người độc lập
để nâng mức tin cậy đánh giá; không dùng kết quả cũ 420 mẫu hoặc Kappa cũ cho bộ test 390 mẫu này.

## 8. Tài liệu tham khảo đã đối chiếu

- [MultinomialNB — scikit-learn 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html).
- [LinearSVC — scikit-learn 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html).
- [TfidfVectorizer — scikit-learn 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

Các điểm số và ví dụ lỗi lấy từ output local của lần chạy, không lấy từ tài liệu tham khảo.
Tài liệu giải thích công thức/FAQ và file code: [part3_learning.md](../docs/part3_learning.md).
