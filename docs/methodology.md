# Phương pháp và lựa chọn triển khai

## Luồng

CSV gốc → validate → group split theo post_id → fit train → chọn Macro-F1 validation →
lưu artifact → evaluate test riêng → compare → phân tích errors.csv.
Baseline dùng nhãn phổ biến nhất train. Không fit vectorizer trước split.
Pipeline tốt nhất giữ nguyên từ train; không refit train+validation trong khung này.

Bộ chia tìm nhóm với seed, tối thiểu hóa sai lệch tỷ lệ dòng và phân bố nhãn, ràng buộc đủ ba nhãn.
Đây là heuristic hữu hạn, không phải nghiệm tối ưu; tỷ lệ dòng xấp xỉ do kích thước nhóm.
Xem tỷ lệ thực tế trong manifest. Chia bài viết giảm rò rỉ ngữ cảnh nhưng không loại hết
rò rỉ từ cùng người viết hay nội dung gần trùng.

## Tiền xử lý truyền thống

| Bước | Lý do | Mặc định |
|---|---|---|
| NFC/khoảng trắng | Đồng nhất biểu diễn ký tự, tránh token rỗng | Bật |
| Lowercase | Giảm biến thể vocabulary | Bật |
| URL → urltoken | Giảm URL riêng nhưng giữ tín hiệu có liên kết | Bật |
| Slang rõ nghĩa | ko/k/khum/hok → không, dc/đc → được, mn → mọi người, sv → sinh viên | Bật, tắt được |
| Emoji | Giữ sắc thái | keep; có remove/map |
| Stop words | Tránh mất ngữ cảnh/phủ định | Danh sách rỗng |
| Underthesea word_tokenize(format='text') | Ghép từ tiếng Việt nhiều âm tiết bằng underscore | Bắt buộc classical |

Trước: “  SV ko  thích lịch học 😢 https://example.org  ”.
Sau normalize mặc định: “sinh viên không thích lịch học 😢 urltoken”.
Sau tách từ, từ nhiều âm tiết như sinh_viên nối underscore.
Đầu ra thực tế phụ thuộc phiên bản, được lưu tại preprocessing_example.json của smoke test.
Stop words khai báo theo token; “sinh viên” đổi thành sinh_viên.
Từ/cụm chứa không/chưa/chẳng/chả/đừng bị từ chối trong stop words.
Không xóa số/dấu câu mặc định. Vectorizer dùng tokenizer khoảng trắng giữ token emoji/dấu câu.
Slang phụ thuộc ngữ cảnh: k có thể là biến toán; cân nhắc tắt expand_slang trong nội dung kỹ thuật.
Không sửa từ điển theo test.

## Đặc trưng/mô hình truyền thống

BoW: CountVectorizer; TF-IDF: TfidfVectorizer (smooth_idf và chuẩn L2 mặc định).
Vocabulary/IDF chỉ học khi pipeline.fit(train).
Thử unigram và unigram+bigram, min_df/max_df/max_features trong YAML.
NB dùng alpha 0.5/1; LinearSVC dùng C 0.5/1, random_state cố định.
Mỗi tổ hợp có 4 ứng viên, chọn Macro-F1 validation; hòa lấy ứng viên trước theo YAML.
Không dùng test chọn tham số. Có thể mở rộng lưới trước thực nghiệm chính thức.
Tham khảo [Pipeline](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html),
[NB](https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html),
[LinearSVC](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html).

## PhoBERT

vinai/phobert-base, Transformers 4.44.2, tối đa 256 token gồm special tokens.
NFC/khoảng trắng → RDRSegmenter VnCoreNLP (chuẩn hóa dấu/tách từ theo công cụ) →
tokenizer use_fast=False → truncation → padding động.
Giữ hoa thường/URL/slang/emoji/stop words, không áp toàn bộ cleaning classical.
Ví dụ “Sinh viên học ở Học viện.” có dạng phân đoạn “Sinh_viên học ở Học_viện .”;
xác nhận đầu ra cụ thể bằng VnCoreNLP thực dùng.
Input phải tách từ và tác giả khuyến nghị RDRSegmenter:
[tài liệu tác giả PhoBERT](https://github.com/VinAIResearch/PhoBERT).

AutoModelForSequenceClassification dùng 3 nhãn từ labels.py.
Trainer đánh giá/lưu mỗi epoch, load_best_model_at_end theo validation eval_macro_f1.
Config chứa batch/learning rate/epoch/weight decay/seed/fp16/use_cpu.
[Trainer 4.44.2](https://huggingface.co/docs/transformers/v4.44.2/main_classes/trainer)
dùng eval_strategy; package ghim phiên bản này.
Inference local_files_only=True, không tải lại checkpoint.
Ghim revision bằng commit trước chạy chính thức; metadata lưu commit nếu có.
VnCoreNLP cần Java/model local, không fallback sang bộ tách từ khác.

## Đánh giá

Accuracy: tỷ lệ đúng; P/R/F1 từng nhãn; support: số mẫu thật.
Macro-P/R/F1 là trung bình không trọng số ba nhãn, hữu ích khi mất cân bằng.
Confusion matrix hàng thật, cột dự đoán. Baseline và mọi mô hình cùng test.
Không suy ra chất lượng tổng quát từ dữ liệu tổng hợp.
Phân tích errors.csv: phủ định, slang, mỉa mai, hỗn hợp, thiếu ngữ cảnh, tách từ, truncation, nhãn nghi vấn.
Không sửa test theo dự đoán để nâng điểm.
