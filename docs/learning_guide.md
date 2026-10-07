# Hướng dẫn học và giải thích dự án

## 1. Từ bình luận đến nhãn

Ví dụ tự viết: “SV ko thích lịch học mới 😢”.
Luồng: kiểm tra CSV → chia bài viết → normalize → tách từ →
vectorizer → classifier → nhãn. Text gốc vẫn được lưu để người phân tích đọc lại.
Normalize mặc định chuyển thành “sinh viên không thích lịch học mới 😢”.
Underthesea có thể ghép sinh_viên, lịch_học. Kết quả tách từ phụ thuộc phiên bản.

Classical tạo vector thưa rồi dùng NB/SVM. PhoBERT dùng RDRSegmenter, tiếp đến
tokenizer subword riêng và mạng Transformer. Cả hai trả cùng một trong ba nhãn.
labels.py là nguồn ánh xạ: positive=0, neutral=1, negative=2.
Sklearn có thể sắp classes_ theo alphabet; khi xuất xác suất code tìm cột theo tên nhãn,
không giả định thứ tự cột sklearn giống ánh xạ PhoBERT.

## 2. Mô hình thực sự học gì?

- Preprocessing theo quy tắc không học nhãn: bảng slang, regex và chính sách emoji do nhóm chọn.
- CountVectorizer học vocabulary và vị trí cột; TfidfVectorizer còn học IDF từ tập train.
- NB học tần suất lớp và phân bố đặc trưng theo lớp, có smoothing alpha.
- SVM học trọng số/hệ số chặn cho các hàm phân biệt lớp, chịu tác động của C.
- PhoBERT đã có trọng số pretrain; fine-tune cập nhật các trọng số và đầu phân loại bằng nhãn của bài toán.
- Baseline chỉ học lớp phổ biến nhất train. Khi hòa, DummyClassifier dùng thứ tự lớp của sklearn.

Không có quy tắc trong code “hễ có từ tốt thì positive”. Từ có ích hay không được suy ra
từ dữ liệu train; dữ liệu đơn giản hoặc thiên lệch sẽ tạo mô hình thiếu khả năng tổng quát.

## 3. fit, transform và predict

fit(X_train, y_train) học trạng thái từ train. Với vectorizer, y không cần thiết.
transform(X_validation) dùng vocabulary/IDF đã học để biến đổi dữ liệu mới;
không thêm từ mới vào vocabulary. Từ chưa thấy có thể không đóng góp đặc trưng.
predict(X_new) dùng mô hình đã lưu để trả nhãn; không gọi fit.

Pipeline.fit gọi fit_transform cho tiền xử lý/vectorizer rồi fit classifier.
Pipeline.predict gọi transform cho các bước đầu rồi predict classifier.
Vì vậy cần lưu cả pipeline: chỉ lưu classifier sẽ thiếu vocabulary, IDF và cấu hình xử lý.

## 4. Train, validation và test

Train là dữ liệu để học; validation chọn alpha/C/ngram hoặc epoch PhoBERT.
Test chỉ đánh giá sau khi chốt các lựa chọn. Nếu nhìn test rồi thay tham số nhiều lần,
test đã trở thành validation và điểm không còn độc lập.

train_classical.py thử từng ứng viên, fit chỉ train, đo Macro-F1 validation.
Nếu hòa lấy ứng viên xuất hiện trước trong YAML. Artifact được chọn vẫn là bản fit train;
không refit train+validation. evaluate.py mới tính test.
Các tệp test được kiểm tra hash/nhãn để xác minh hợp đồng dữ liệu;
nội dung test không được dùng học tham số hoặc chọn ứng viên.

## 5. Vì sao rò rỉ dữ liệu nguy hiểm?

Hai bình luận cùng bài viết thường chung chủ đề/từ khóa. Nếu một bài ở cả train/test,
mô hình có thể nhận ra ngữ cảnh đã thấy thay vì tổng quát.
Một câu trùng cả hai tập làm test quá dễ. Fit IDF trên toàn dữ liệu cũng cho train biết
phân bố từ của test. Các rò rỉ này có thể làm điểm cao hơn khả năng triển khai thực.

split.py chia nguyên post_id, check_disjoint kiểm tra id/post_id/khóa text.
Khóa text dùng NFC/casefold/gộp khoảng trắng; không đảm bảo phát hiện paraphrase.
Manifest giữ hash và thành viên. Không chọn seed dựa theo kết quả test.
Việc che email/số điện thoại trong preprocessing không làm dữ liệu nguồn hết nhạy cảm:
predictions/errors phải giữ text theo hợp đồng và cần được nhóm bảo quản, rà ẩn danh trước chia sẻ.

## 6. Ví dụ BoW và TF-IDF

Đây là phép tính minh họa, không phải dữ liệu/thực nghiệm PTIT:
d1 = “học tốt”, d2 = “học vui”, d3 = “không tốt”.
Giả sử tách token đúng như khoảng trắng và vocabulary [học, tốt, vui, không].

| Câu | học | tốt | vui | không |
|---|---:|---:|---:|---:|
| d1 | 1 | 1 | 0 | 0 |
| d2 | 1 | 0 | 1 | 0 |
| d3 | 0 | 1 | 0 | 1 |

BoW đếm số lần xuất hiện. Nó không giữ thứ tự ngoài n-gram.
Bigram thêm đặc trưng “không tốt”, giúp phân biệt với “tốt”.

Với smooth_idf mặc định sklearn: idf(t)=ln((1+n)/(1+df(t)))+1.
n=3; học/tốt xuất hiện ở 2 câu nên IDF≈1.288; vui/không xuất hiện ở 1 câu nên IDF≈1.693.
TF-IDF trước chuẩn hóa cho d2 là [1.288,0,1.693,0].
Sau chuẩn L2, chia cho sqrt(1.288²+1.693²), xấp xỉ [0.605,0,0.796,0].
Từ hiếm hơn có trọng số IDF cao hơn, nhưng không đồng nghĩa quan trọng hơn về cảm xúc.
Xem features/vectorizers.py và [sklearn feature extraction](https://scikit-learn.org/1.5/modules/feature_extraction.html).

## 7. Phần 3: Multinomial Naive Bayes

NB chọn lớp tối đa hóa log P(c) + tổng_j x_j log P(t_j|c).
P(c) lấy từ tỷ lệ nhãn train. P(t|c) ước lượng từ số đếm đặc trưng của lớp,
cộng alpha để không có xác suất 0 cho từ chưa gặp ở lớp đó.
Giả định “naive” là đặc trưng độc lập có điều kiện theo lớp; ngôn ngữ thực không hoàn toàn thỏa giả định.

Ví dụ giả định có 3 từ [tốt, tệ, học]. Tổng số đếm lớp positive là [4,0,2],
negative [0,4,2], neutral [1,1,4], prior ba lớp bằng nhau.
Với alpha=1, P(tốt|positive)=5/9, P(tốt|negative)=1/9,
P(tốt|neutral)=2/9. Câu chỉ có “tốt” nghiêng positive.
Đây là ví dụ tính tay, không phải tham số của artifact thật.

Alpha lớn làm phân bố trơn hơn. NB+TF-IDF vẫn được dùng với đặc trưng không âm,
dù trọng số TF-IDF không còn là số đếm nguyên theo mô hình sinh ban đầu.
Không giải thích predict_proba của NB như xác suất đã hiệu chỉnh.
Code: models/classical.py tạo MultinomialNB; training/train_classical.py chọn alpha bằng validation.

## 8. Phần 3: SVM tuyến tính

Hãy tưởng tượng hai trục “từ khen” và “từ chê”. SVM tìm ranh giới phân biệt các lớp,
ưu tiên margin lớn đồng thời phạt lỗi phân loại.
LinearSVC mặc định xử lý đa lớp kiểu one-vs-rest: mỗi lớp có hàm điểm tuyến tính w·x+b,
rồi lấy lớp có điểm cao nhất. Code sử dụng squared hinge loss mặc định.
C lớn tăng mức phạt vi phạm, có thể khớp train hơn; C nhỏ tăng tác dụng regularization.
Tác động thực phải đo trên validation, không khẳng định C lớn luôn tốt hơn.

Decision score có thể âm hoặc lớn hơn 1, không phải probability.
Không đưa decision_function qua một phép đổi tên rồi báo xác suất.
Code: models/classical.py tạo LinearSVC; predict_classical kiểm tra khả năng predict_proba.

## 9. Pretrain và fine-tune

Pretrain giúp mô hình học biểu diễn ngôn ngữ từ lượng văn bản lớn bằng nhiệm vụ tự giám sát.
Fine-tune sử dụng dữ liệu nhãn cụ thể để tối ưu phân loại cảm xúc.
PhoBERT cần tách từ trước BPE; RDRSegmenter và tokenizer subword giải quyết hai cấp khác nhau.
Ví dụ từ nhiều âm tiết có underscore rồi được chia thành các mảnh trong vocabulary của checkpoint.

train_phobert.py dùng AutoModelForSequenceClassification, padding động, tối đa 256 token gồm special tokens.
Cắt phía phải: phần cuối bình luận có thể mất, kể cả vế “nhưng ...”.
preprocessing/phobert_input.py đếm token trước cắt; truncation.json ghi số lượng/tỷ lệ/id bị cắt
trên train và validation. Evaluate lưu thống kê test trong metrics.json.
Chọn checkpoint bằng Macro-F1 validation; config smoke 1 bước chỉ thử cơ chế chạy.
Chưa chạy model thật thì không được mô tả như đã fine-tune thành công.

## 10. Tính độ đo cho ba nhãn

Ví dụ tính tay (không phải kết quả dự án), hàng thật/cột dự đoán theo positive,neutral,negative:

| Thật / Dự đoán | positive | neutral | negative | Support |
|---|---:|---:|---:|---:|
| positive | 3 | 1 | 0 | 4 |
| neutral | 1 | 2 | 1 | 4 |
| negative | 0 | 1 | 3 | 4 |

Accuracy=(3+2+3)/12=2/3≈0.667.
Với positive: TP=3, FP=1, FN=1 nên P=3/4, R=3/4, F1=0.75.
Neutral: P=2/4, R=2/4, F1=0.5. Negative: P=R=F1=0.75.
Macro-P=Macro-R=Macro-F1=(0.75+0.5+0.75)/3≈0.667.
Support là số nhãn thật, không phải số dự đoán của lớp.
Thông thường Macro-F1 không bằng F1 tính từ Macro-P và Macro-R:
code tính F1 từng lớp rồi lấy trung bình.
Nếu không dự đoán một lớp, precision lớp đó được đặt 0 theo zero_division=0.

## 11. Đọc confusion matrix và phân tích lỗi

Đường chéo là dự đoán đúng. Ô hàng negative/cột neutral lớn nghĩa là nhiều câu tiêu cực
bị làm nhẹ thành trung tính; không phải ngược lại.
Dùng errors.csv để đọc câu, nhãn thật và dự đoán, rồi gắn nhóm nguyên nhân:
phủ định, mỉa mai, đa sắc thái, viết tắt, thiếu ngữ cảnh, tách từ, truncation hoặc nhãn đáng xem lại.
Một nhận xét phải dẫn id ví dụ thật hoặc thống kê đã đếm.
Không sửa test theo dự đoán để nâng điểm; nếu sửa nhãn do lỗi đã được xác minh,
tạo phiên bản dữ liệu mới và báo rõ việc thay đổi.

## 12. Lộ trình đọc code và tự giải thích

| Bước | File dưới src/ptit_sentiment/ | Điều cần giải thích |
|---|---|---|
| Schema/nhãn | labels.py, data/validate.py | Nhãn nằm ở đâu, dữ liệu sai bị chặn thế nào? |
| Chia tập | data/split.py | Nhóm bài viết và hash chống rò rỉ ra sao? |
| Xử lý | preprocessing/normalize.py, tokenize.py | Vì sao giữ phủ định, emoji? |
| Vector | features/vectorizers.py | Vocabulary/IDF học khi nào? |
| NB/SVM | models/classical.py | alpha, C, score và probability? |
| Chọn cấu hình | training/train_classical.py | train/validation/test dùng ở đâu? |
| PhoBERT | training/train_phobert.py, models/phobert.py | Chọn checkpoint, cắt token, thiết bị? |
| Đánh giá | evaluation/metrics.py, evaluate.py, compare.py | Macro, support, trục ma trận? |
| Dự đoán | predict.py | Vì sao không cần fit lại? |

Người phụ trách phần 3 nên đọc mục 3–8, 10–11 của tài liệu này, rồi lần lượt đọc
models/classical.py → training/train_classical.py → evaluation/metrics.py → evaluate.py → compare.py.
Sau đó mở report/report.md mục 5–7 và tự lần theo từng số về metrics.json/training log.
Bài tập tự kiểm tra: giải thích vì sao baseline trong lần minh họa chỉ dự đoán một lớp,
tại sao bốn mô hình đạt điểm cao trên template nhưng chưa chứng minh hiệu quả trên Facebook thật.
