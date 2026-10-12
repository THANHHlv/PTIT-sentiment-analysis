# Tự giải thích phần 3: NB, SVM và đánh giá

## Từ bình luận đến nhãn

Pipeline nhận **text gốc** → chuẩn hóa NFC/khoảng trắng, che thông tin nhạy cảm,
thay URL, mở rộng slang → Underthesea tách từ → BoW/TF-IDF → classifier → nhãn.
Ví dụ minh họa `SV ko thích lịch học` thành `sinh_viên không thích lịch_học`.
Một cột vector ứng với một token hoặc cặp token liên tiếp có trong vocabulary train.
CSV gốc không bị sửa. Pipeline đã lưu chứa cả ba bước; không tiền xử lý lần nữa trước `predict`.

Code: [tiền xử lý](../src/ptit_sentiment/preprocessing/tokenize.py),
[vectorizer](../src/ptit_sentiment/features/vectorizers.py),
[pipeline](../src/ptit_sentiment/models/classical.py).

## Fit, transform và predict

| Thao tác | Học gì / trả gì? | Dùng ở đâu? |
|---|---|---|
| `TextPreprocessor.fit` | Kiểm tra cấu hình; không học từ bình luận | Train |
| `vectorizer.fit_transform` | Học vocabulary, DF/IDF và trả ma trận | Chỉ train |
| `vectorizer.transform` | Dùng vocabulary/IDF đã học, không thêm cột | Validation, test, câu mới |
| `classifier.fit` | Học tham số phân loại từ vector và nhãn | Chỉ train |
| `pipeline.predict` | Xử lý text gốc rồi trả nhãn, không fit lại | Validation, test, câu mới |

Validation chọn cấu hình bằng Macro-F1. Test đo kết quả sau khi đã chốt.
Code [train_classical.py](../src/ptit_sentiment/training/train_classical.py) tách từ train/validation
một lần bằng bộ xử lý không học tham số. Mỗi ứng viên vẫn có vectorizer và classifier riêng;
pipeline hoàn chỉnh được ghép lại để nhận text gốc. Test không tham gia fit/chọn tham số.

## BoW, TF-IDF, unigram và bigram

BoW dùng số đếm, ví dụ `tốt tốt` có giá trị 2 ở cột `tốt`.
TF-IDF mặc định dùng `tf × [log((1+n)/(1+df))+1]`, sau đó chuẩn hóa L2;
`n` và `df` chỉ tính trên train. Cả hai đều là ma trận sparse không âm nên dùng được với MultinomialNB.
Unigram dùng token riêng; unigram–bigram thêm cặp như `không tốt` để giữ chút ngữ cảnh cục bộ.
Bigram không hiểu được toàn bộ cú pháp và tăng số chiều. `min_df=1` giữ từ hiếm;
`max_df=1.0` không cắt từ phổ biến; `max_features=20000` đặt trần số cột.
Các giá trị này được giữ từ NV2 vì train có 1.801 câu, không đổi theo điểm test.
Tham khảo [TfidfVectorizer 1.5](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

## Multinomial Naive Bayes học gì?

NB học prior của lớp `P(c)` và trọng số token theo lớp. Với vector `x`, điểm lớp là:

```text
score(c) = log P(c) + Σ_j x_j log θ_cj
θ_cj = (N_cj + alpha) / (Σ_k N_ck + alpha × V)
```

`N_cj` là tổng giá trị đặc trưng j của các câu train mang nhãn c; `V` là số cột.
Chọn lớp có điểm cao nhất. Dùng log để tránh nhân nhiều xác suất nhỏ.
Giả định độc lập có điều kiện giúp mô hình đơn giản nhưng không phản ánh đầy đủ quan hệ giữa các từ.
Alpha thêm lượng giả định vào mỗi token để tránh xác suất 0; alpha lớn làm phân bố token bớt cực đoan.
Với TF-IDF, các tổng là trọng số thực không âm; đây là ứng dụng thực dụng của NB,
không phải số đếm từ nguyên như giả định Multinomial ban đầu.
Tham khảo [MultinomialNB 1.5](https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

## LinearSVC phân loại ba lớp thế nào?

Mặc định `multi_class='ovr'`: học ba bài toán một lớp so với hai lớp còn lại.
Mỗi lớp có vector trọng số `w_c` và hệ số chặn `b_c`; dự đoán lấy lớp có
`w_c · x + b_c` lớn nhất. `C` cân bằng phạt lỗi vi phạm lề và độ lớn trọng số.
C lớn phạt vi phạm mạnh hơn, có thể khớp train hơn; không đảm bảo tốt hơn trên dữ liệu mới.
Lần này dùng L2, squared hinge, `dual='auto'`, `tol=1e-4`, seed 42.
SVM không xuất xác suất. Decision score không phải độ mạnh cảm xúc.
Tham khảo [LinearSVC 1.5](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html).

## Đọc các độ đo

Với từng lớp c, coi c là dương và các lớp còn lại là âm:

```text
Accuracy = số dự đoán đúng / tổng số câu
Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
F1_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)
Macro-Precision = (Precision_positive + Precision_neutral + Precision_negative) / 3
Macro-Recall = (Recall_positive + Recall_neutral + Recall_negative) / 3
Macro-F1 = (F1_positive + F1_neutral + F1_negative) / 3
```

Mẫu số 0 được xử lý bằng `zero_division=0`. Macro-F1 không phải F1 tính từ Macro-Precision/Recall.
Support là số nhãn thật của mỗi lớp. Accuracy dễ cao khi neutral chiếm đa số;
Macro-F1 cho mỗi lớp cùng trọng số nên được dùng chọn cấu hình.
Code chung: [metrics.py](../src/ptit_sentiment/evaluation/metrics.py).

Ví dụ **minh họa**, không phải kết quả PTIT:

| Nhãn thật \ dự đoán | positive | neutral | negative |
|---|---:|---:|---:|
| positive | 8 | 2 | 0 |
| neutral | 1 | 17 | 2 |
| negative | 0 | 3 | 7 |

Đường chéo có 32 câu đúng trên 40: Accuracy=0,8.
Positive có TP=8, FP=1, FN=2: Precision=8/9, Recall=8/10, F1=16/19.
Ô hàng negative/cột neutral = 3 nghĩa là ba câu tiêu cực bị đoán trung tính.
Luôn đọc hàng là thật, cột là dự đoán theo thứ tự `positive, neutral, negative`.

## Câu hỏi khi bảo vệ

| Câu hỏi | Trả lời ngắn |
|---|---|
| Tại sao không fit TF-IDF trên cả dữ liệu? | Sẽ đưa thông tin tập đánh giá vào vocabulary/IDF trước khi đo. |
| Tại sao không train lại lúc predict? | Cần dùng đúng tham số đã chọn và lưu để kết quả có thể truy vết. |
| Câu có toàn từ mới thì sao? | Vector 0; NB dựa vào prior, SVM dựa vào intercept; không có nghĩa câu trung tính. |
| Câu rỗng thì sao? | Pipeline/CLI báo lỗi rõ; không tự gán neutral. |
| Baseline học gì? | Chỉ học lớp xuất hiện nhiều nhất trong train: neutral. |
| Tại sao giữ phủ định, emoji? | Chúng mang tín hiệu cảm xúc; loại chúng có thể đảo nghĩa câu. |
| Tại sao cấu hình validation tốt nhất chưa chắc test tốt nhất? | Hai tập có mẫu/bài khác nhau; validation chỉ là ước lượng để lựa chọn. |
| Test AI chứng minh được gì? | Mức phù hợp với nhãn AI đã khóa; chưa chứng minh chất lượng trên nhãn người độc lập. |
| Có thể đổi C sau khi đọc lỗi test? | Đó là thử nghiệm mới có sử dụng test; không tiếp tục gọi điểm cũ là đánh giá độc lập. |
| Phần 3 đã bao gồm PhoBERT chưa? | Chưa; người 4 cần dùng cùng manifest và hàm đánh giá chung. |

Xem kết quả thật, cấu hình được chọn và phân tích lỗi ở
[báo cáo phần 3](../report/part3_classical.md), cách dùng ở [bàn giao](part3_handoff.md).
