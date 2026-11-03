# Người 3 — Huấn luyện và đánh giá mô hình truyền thống

Đề tài: **Xây dựng hệ thống phân loại cảm xúc bình luận của sinh viên PTIT trên các nhóm Facebook**.
Lần huấn luyện thực tế: **part3_final_20261102**, ngày 02/11/2026, CPU.
Đã chạy lại demo và đánh giá bốn pipeline để tổng hợp báo cáo ngày 02/11/2026.
Phạm vi: bốn tổ hợp BoW/TF-IDF × MultinomialNB/LinearSVC và baseline đối chiếu.
Không thu thập, gán lại nhãn, chia lại dữ liệu, viết lại tiền xử lý hoặc train PhoBERT.

TF-IDF + SVM được chọn bằng Macro-F1 validation (0,577487). Trên 390 câu test,
mô hình đạt Accuracy 71,79% và Macro-F1 0,5354. Hai lớp positive/negative còn
khó nhận diện; kết quả demo cũng cho thấy những dự đoán chưa phù hợp sắc thái.
Vì vậy, kết quả cần được diễn giải trong phạm vi bộ nhãn AI và test đã sử dụng.

## 1. Đầu vào và nguồn nhãn

Dùng trực tiếp bộ dữ liệu/bộ chia do người 1 và code tiền xử lý/đặc trưng do người 2 bàn giao.
Dataset ai_initial_4e93595ef7d1_seed42, seed 42.

| Tập | Số câu | Số bài | positive | neutral | negative |
|---|---:|---:|---:|---:|---:|
| Train | 1801 | 326 | 254 | 1246 | 301 |
| Validation | 409 | 63 | 58 | 283 | 68 |
| Test | 390 | 57 | 50 | 271 | 69 |
| Tổng | 2600 | 446 | 362 | 1800 | 438 |

Tất cả nhãn vẫn có label_source=ai, annotation_status=ai_labeled; human_labels_confirmed=false.
Đây là nhãn tham chiếu AI của bộ bàn giao, chưa phải nhãn người độc lập. Excel review trong
outputs/label_review/ được giữ riêng, chưa ghép vào lần chạy này. Test đã được dùng ở các lần
trước; lần hiện hành là so sánh bổ sung trên cùng test, không phải kiểm chứng độc lập mới.

Manifest SHA-256: c743878f73ef20ef786f46f46c62511b894a10891c2893de8d75cb753522e4e3.
Test SHA-256: afcb8ce871c14ea258fc02a8ec56596a2c7e2e034b88d2437f2cdffdc0184312.
Kiểm tra ID/bài/nội dung chuẩn hóa và ràng buộc gần trùng theo manifest đạt; không đổi hash để vượt kiểm tra.

## 2. Cơ sở mô hình

### Multinomial Naive Bayes

Với vector không âm x, mô hình chọn lớp có điểm log P(c) + Σ_j x_j log θ_cj lớn nhất.
θ_cj = (N_cj + alpha)/(Σ_k N_ck + alpha × V), với V là số đặc trưng, N_cj là tổng
giá trị đặc trưng j trong lớp c của train. Alpha làm trơn để tránh xác suất bằng 0;
alpha lớn làm phân bố token bớt cực đoan. Giả định độc lập có điều kiện giúp mô hình
đơn giản, nhưng hạn chế biểu diễn quan hệ giữa từ. BoW dùng số đếm; TF-IDF không âm
vẫn có thể dùng thực dụng với NB dù không còn số đếm nguyên.
[Tài liệu MultinomialNB 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

### SVM tuyến tính

LinearSVC học ba hàm one-vs-rest w_c·x+b_c và chọn lớp có điểm cao nhất.
C điều chỉnh mức phạt vi phạm lề so với điều chuẩn; C lớn không bảo đảm tốt hơn
trên dữ liệu mới. Dùng L2, squared hinge, dual=auto, tol=1e-4, max_iter=10000, seed 42.
Code bắt cảnh báo hội tụ, thử lại tối đa một lần với max_iter=100000 rồi báo lỗi nếu vẫn chưa hội tụ.
class_weight=None được chốt trong lần này. Nếu dùng balanced trong nghiên cứu khác, trọng số
lớp bằng n_train/(3 × số mẫu lớp), chỉ tính từ train. Không thử thêm sau khi xem test.
Decision score của SVM không phải xác suất hoặc độ mạnh cảm xúc.
[Tài liệu LinearSVC 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html).

## 3. Pipeline và thiết kế thực nghiệm

Text gốc → TextPreprocessor của người 2 → vectorizer → classifier. Pipeline joblib
chứa đủ ba bước, nhận text gốc khi predict. Giữ nguyên NFC/khoảng trắng, che thông tin
nhạy cảm, thay URL, slang map ko/khum/hok, Underthesea, emoji keep và stop words rỗng.
Không thay luật tiền xử lý theo lỗi test.

BoW dùng CountVectorizer; TF-IDF dùng TfidfVectorizer, smooth IDF, chuẩn L2.
min_df=1, max_df=1.0, max_features=20000. Vocabulary/DF/IDF chỉ học train; validation/test
chỉ transform. Tiền xử lý theo quy tắc được cache cho cùng chế độ emoji; mỗi ứng viên
vẫn có vectorizer/classifier riêng. Không fit trên train+validation.
Cấu trúc này theo [Pipeline của scikit-learn](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html).

[part3.yaml](../configs/part3.yaml) chốt trước lần đánh giá: word n-gram (1,1)/(1,2);
NB alpha=0.5/1.0; SVM C=0.5/1.0; class_weight=None. Mỗi tổ hợp có 4 ứng viên, tổng 16.
Chọn Macro-F1 validation, hòa lấy cấu hình đầu YAML. Không dùng nhánh word-char trong lần này.
Baseline học lớp phổ biến nhất từ train: neutral.

| Mô hình | N-gram chọn | Tham số chọn | Macro-F1 validation |
|---|---|---|---:|
| BoW + NB | (1,1) | alpha=0.5 | 0.561527 |
| TF-IDF + NB | (1,1) | alpha=0.5 | 0.344149 |
| BoW + SVM | (1,1) | C=0.5 | 0.566607 |
| TF-IDF + SVM | (1,1) | C=1 | 0.577487 |

**TF-IDF + SVM là mô hình chính**, được chọn bằng validation trước khi xem kết quả test
của lần chạy này. Các khóa selection_lock.json và all_selection_lock.json lưu thời điểm,
hash cấu hình và artifact trước evaluate test. Không đổi lựa chọn sau khi đọc test.

## 4. Chỉ số đánh giá

Với từng nhãn, Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2PR/(P+R); support là số mẫu thật.
Accuracy là tổng đường chéo confusion matrix chia số mẫu. Macro-P/R/F1 là trung bình
không trọng số của từng chỉ số qua ba nhãn; Macro-F1 không phải F1 tính từ Macro-P/R.
Dùng zero_division=0 khi mẫu số bằng 0. Macro-F1 giúp thấy chất lượng hai lớp ít mẫu
thay vì để neutral chi phối như Accuracy. Hàm đánh giá dùng chung đủ ba nhãn
positive/neutral/negative cho cả NB, SVM và nhãn dự đoán PhoBERT.
[Tài liệu precision_recall_fscore_support](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).

## 5. Kết quả thực tế trên 390 câu test

Nguồn: outputs/part3_final_20261102/comparison_core.csv và evaluation/*/metrics.json.

| Mô hình | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Số sai |
|---|---:|---:|---:|---:|---:|
| BoW + NB | 0.715385 | 0.570452 | 0.476760 | 0.498018 | 111 |
| TF-IDF + NB | 0.702564 | 0.457209 | 0.351427 | 0.310785 | 116 |
| BoW + SVM | 0.705128 | 0.548639 | 0.513427 | 0.526752 | 115 |
| TF-IDF + SVM | 0.717949 | 0.571906 | 0.515976 | 0.535355 | 110 |

![So sánh bốn mô hình](../outputs/part3_final_20261102/comparison_core.png)

Baseline neutral: Accuracy=0.694872,
Macro-F1=0.273323, F1 positive/negative=0.
Test có 271/390 neutral (69,49%), nên Accuracy gần 70% chưa chứng minh phân loại tốt cả ba lớp.

| Mô hình | Nhãn | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| BoW + NB | positive | 0.476190 | 0.200000 | 0.281690 | 50 |
| BoW + NB | neutral | 0.767081 | 0.911439 | 0.833052 | 271 |
| BoW + NB | negative | 0.468085 | 0.318841 | 0.379310 | 69 |
| TF-IDF + NB | positive | 0.000000 | 0.000000 | 0.000000 | 50 |
| TF-IDF + NB | neutral | 0.704961 | 0.996310 | 0.825688 | 271 |
| TF-IDF + NB | negative | 0.666667 | 0.057971 | 0.106667 | 69 |
| BoW + SVM | positive | 0.416667 | 0.300000 | 0.348837 | 50 |
| BoW + SVM | neutral | 0.795918 | 0.863469 | 0.828319 | 271 |
| BoW + SVM | negative | 0.433333 | 0.376812 | 0.403101 | 69 |
| TF-IDF + SVM | positive | 0.454545 | 0.300000 | 0.361446 | 50 |
| TF-IDF + SVM | neutral | 0.789474 | 0.885609 | 0.834783 | 271 |
| TF-IDF + SVM | negative | 0.471698 | 0.362319 | 0.409836 | 69 |

TF-IDF + NB có recall positive=0, negative≈0,058 và neutral≈0,996.
Giả thuyết cần kiểm tra: TF-IDF L2 làm giảm tổng trọng số đặc trưng và, với alpha đã chốt,
prior/smoothing có thể chi phối hơn tín hiệu của các lớp ít mẫu. Kết quả không đủ chứng minh
đây là nguyên nhân duy nhất; không điều chỉnh alpha theo test này.

## 6. Confusion Matrix

Tất cả ma trận có **hàng là nhãn thật (AI), cột là nhãn dự đoán**,
thứ tự positive/neutral/negative. Tổng hàng là support; tổng toàn ma trận là 390.

### BoW + NB

| Nhãn thật / dự đoán | positive | neutral | negative |
|---|---:|---:|---:|
| positive | 10 | 32 | 8 |
| neutral | 7 | 247 | 17 |
| negative | 4 | 43 | 22 |

![Confusion matrix BoW + NB](../outputs/part3_final_20261102/evaluation/bow_nb/confusion_matrix.png)

### TF-IDF + NB

| Nhãn thật / dự đoán | positive | neutral | negative |
|---|---:|---:|---:|
| positive | 0 | 48 | 2 |
| neutral | 1 | 270 | 0 |
| negative | 0 | 65 | 4 |

![Confusion matrix TF-IDF + NB](../outputs/part3_final_20261102/evaluation/tfidf_nb/confusion_matrix.png)

### BoW + SVM

| Nhãn thật / dự đoán | positive | neutral | negative |
|---|---:|---:|---:|
| positive | 15 | 25 | 10 |
| neutral | 13 | 234 | 24 |
| negative | 8 | 35 | 26 |

![Confusion matrix BoW + SVM](../outputs/part3_final_20261102/evaluation/bow_svm/confusion_matrix.png)

### TF-IDF + SVM

| Nhãn thật / dự đoán | positive | neutral | negative |
|---|---:|---:|---:|
| positive | 15 | 27 | 8 |
| neutral | 11 | 240 | 20 |
| negative | 7 | 37 | 25 |

![Confusion matrix TF-IDF + SVM](../outputs/part3_final_20261102/evaluation/tfidf_svm/confusion_matrix.png)

## 7. So sánh đặc trưng và thuật toán

- Giữ NB: BoW có Macro-F1 cao hơn TF-IDF 0.187233;
  cả hai chọn alpha=0.5, unigram. BoW + NB nhận diện hai lớp thiểu số tốt hơn trong bộ này.
- Giữ SVM: TF-IDF cao hơn BoW 0.008603.
  Hai mô hình chọn C khác nhau (1 so với 0.5), nên đây là so sánh hai pipeline được chọn
  bằng validation, không phải thí nghiệm chỉ thay duy nhất biểu diễn.
- Giữ BoW: SVM cao hơn NB 0.028735 về Macro-F1,
  dù Accuracy thấp hơn. Giữ TF-IDF: SVM cao hơn NB 0.224570.
- TF-IDF + SVM đứng đầu cả validation và test trong bốn mô hình của lần này; chưa
  kiểm định ý nghĩa thống kê và không kết luận tốt nhất cho mọi dữ liệu.

Mô hình chính nhầm positive→neutral 27 câu, negative→neutral 37 câu, tổng 64/110 lỗi
(58,18%). Recall positive=0,30 và negative≈0,362 còn thấp dù Accuracy=0,718.
Đây là quan sát phù hợp với vấn đề mất cân bằng; chưa tách được tác động riêng của
phân bố nhãn, chất lượng nhãn AI và biểu diễn đặc trưng.

## 8. Phân tích lỗi thật

Đã đọc errors.csv của TF-IDF + SVM và chọn 25 trích đoạn thật bao phủ sáu hướng nhầm.
Mỗi ví dụ có mã, nội dung, nhãn tham chiếu, nhãn dự đoán, quan sát và giả thuyết trong
[phân tích lỗi](part3_error_analysis.md). File local error_examples.csv có thêm ID gốc
để người 3 đối soát; ánh xạ/hash trong error_example_links_local.json không đưa vào gói công khai.
Mẫu định tính có chủ đích, không đại diện cho tỷ lệ nguyên nhân của toàn bộ lỗi.
Không viết lại nội dung gốc, đổi nhãn hoặc tuning sau phân tích.

### Ba trường hợp tiêu biểu

**E01 - “Vài câu đề k rõ ràng và bị lỗi testcase”**

Nhãn tham chiếu AI: negative; dự đoán: neutral.
Quan sát: Nêu đề không rõ và lỗi testcase, dùng viết tắt k.
Giả thuyết cần kiểm tra: Từ viết tắt và từ nội dung kỹ thuật có thể làm tín hiệu chê yếu; cần xem trọng số mới xác nhận.

**E10 - “đã vote 1 sao, xin cảm ơn.”**

Nhãn tham chiếu AI: negative; dự đoán: positive.
Quan sát: Có đánh giá 1 sao và lời cảm ơn trong cùng câu.
Giả thuyết cần kiểm tra: Lời lịch sự có thể lấn tín hiệu chê; đây là giả thuyết về mỉa mai.

**E24 - “Không gì là không thể e ạ”**

Nhãn tham chiếu AI: positive; dự đoán: neutral.
Quan sát: Động viên bằng hai phủ định.
Giả thuyết cần kiểm tra: Unigram khó biểu diễn phủ định kép và ý khích lệ.

## 9. Chạy demo bằng pipeline đã huấn luyện

Đã chạy trực tiếp CLI cho cả bốn pipeline, gồm một file CSV ba câu minh họa
và một bình luận nhập bằng --text cho mỗi model: tổng cộng 8 lệnh dự đoán.
Đồng thời chạy lại evaluate trên cùng 390 mẫu test cho bốn model; toàn bộ 1.560
dự đoán, các chỉ số, support và confusion matrix khớp lần đánh giá đã lưu.
Hash dữ liệu, artifact, code NV2 và Excel review giữ nguyên; không huấn luyện lại.

Các câu dưới đây do nhóm tự viết để minh họa thao tác, không phải dữ liệu test
và không có nhãn chuẩn để tính Accuracy. Giữ nguyên dự đoán thực tế, không lựa
chọn lại câu hoặc sửa output để tạo cảm giác mô hình luôn đúng.

- **new-01:** “Thầy giải thích bài rất dễ hiểu, mình rất hài lòng.”
- **new-02:** “Lịch thi được đăng vào thứ hai.”
- **new-03:** “Cổng đăng ký lỗi mãi, mình rất thất vọng.”
- **demo-04:** “Thầy giảng rất dễ hiểu.”

| Mẫu | BoW + NB | TF-IDF + NB | BoW + SVM | TF-IDF + SVM |
|---|---|---|---|---|
| new-01 | negative | neutral | positive | negative |
| new-02 | neutral | neutral | neutral | neutral |
| new-03 | negative | neutral | positive | positive |
| demo-04 | positive | neutral | negative | negative |

Mẫu new-02 cho kết quả neutral ở cả bốn model. Mẫu new-01 có các cụm
“rất dễ hiểu”, “rất hài lòng”, nhưng BoW + NB và TF-IDF + SVM dự đoán negative.
Mẫu new-03 có “rất thất vọng”, song hai SVM dự đoán positive. Mẫu demo-04
cũng cho dự đoán khác nhau giữa các thuật toán. Đây là những trường hợp cần
thận trọng khi sử dụng, không phải bằng chứng demo thành công về ngữ nghĩa.

Kết quả cho thấy việc chạy đúng pipeline không bảo đảm hiểu đúng mọi câu.
Các mô hình học liên hệ thống kê từ bộ nhãn AI nhỏ và lệch lớp; việc giải thích
chính xác từng dự đoán cần xem trọng số/đặc trưng và đánh giá bằng nhãn người.
Chưa thực hiện can thiệp để xác định nguyên nhân riêng của từng trường hợp.
Không điều chỉnh model sau khi đọc các câu demo này.

Lệnh PowerShell tại thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.predict `
  --model artifacts/part3_final_20261102/tfidf_svm.joblib `
  --text "Thầy giảng rất dễ hiểu."
```

Bảng demo được lưu tại outputs/part3_final_20261102/demo/predictions.csv.
Biên bản chạy, argv/stdout và đối soát test ở demo/verification.json;
verified_test_metrics.csv chứa bảng chỉ số vừa kiểm chứng lại.

## 10. Kiểm chứng và bàn giao

68 kiểm thử đã đạt trong đợt hoàn thiện code; lần tổng hợp báo cáo này chạy lại
demo và đánh giá test, không tìm lại tham số hoặc huấn luyện lại.
Các kiểm thử bao gồm kiểm tra giao dữ liệu, nguồn nhãn, pipeline chỉ fit train,
lưu/tải, hội tụ SVM, hỗ trợ ba nhãn và xuất metric cho dự đoán PhoBERT độc lập artifact.
Audit đối chiếu vocabulary/IDF từ train và giữ nguyên tham số khi transform heldout.
Pipeline tải lại giữ nguyên toàn bộ dự đoán test; metrics tính lại, support, confusion
matrix, errors.csv và ID/thứ tự/text đều khớp. CLI một câu/CSV, câu rỗng và artifact thiếu
được kiểm tra thực tế. Hash dữ liệu, manifest, workbook review và code NV2 được bảo toàn.

Artifact ở artifacts/part3_final_20261102/. Output ở outputs/part3_final_20261102/.
Lệnh tái lập và sử dụng ở [README](../README.md); API/phiên bản test cho người 4 ở
[bàn giao](../docs/part3_handoff.md); lý thuyết bảo vệ ở [tài liệu học](../docs/part3_learning.md).
Không train lại lúc predict. PhoBERT chưa có kết quả trong lần này; chỉ ghép output thật
trên đúng manifest/test khi người 4 cung cấp.

## 11. Hạn chế và kết luận

Đã hoàn thành phần người 3 trong điều kiện bộ bàn giao: bốn pipeline thật, lựa chọn
validation, đánh giá test, baseline và phân tích lỗi có truy vết. Lưới nhỏ và một bộ chia
chưa bao quát mọi tham số. Nhãn AI, test từng dùng, lệch lớp, câu thiếu ngữ cảnh, slang,
phủ định và mỉa mai hạn chế diễn giải chất lượng. Cần test mới được người gán độc lập
xác nhận để đánh giá khả năng khái quát. Không suy ra chất lượng toàn đề tài hoặc
so sánh với PhoBERT khi chưa có thực nghiệm của người 4.

## Tài liệu tham khảo

Đã đối chiếu tài liệu chính thức scikit-learn 1.5.2 ngày 02/11/2026:

1. [MultinomialNB](https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html).
2. [LinearSVC](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html).
3. [Pipeline](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html).
4. [Precision, Recall, F-score và support](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).
