# AGENTS.md — PTIT Sentiment Analysis

## Chỉ dẫn dọn dữ liệu — 07/10/2026

Theo yêu cầu mới nhất, data chỉ giữ labeled_comments.csv, train.csv,
validation.csv, test.csv và manifest.json chứa metadata nhúng. Raw,
unresolved/excluded, hồ sơ kiểm toán/gán nhãn được lưu riêng trong
outputs/archives/data_minimal_20261007.zip; không tự đưa lại vào data.
Dữ liệu tổng hợp để kiểm tra code đã chuyển sang tests/fixtures/.
Các quy định bên dưới về vị trí lưu nguồn/fixture áp dụng trước khi dọn;
bản gốc trong ZIP vẫn giữ nguyên. Không thay nguồn nhãn AI hoặc huấn luyện.

## Chỉ dẫn cập nhật của người dùng — bộ nhãn AI ban đầu

Theo yêu cầu hiện tại, cho phép phân tích ngôn ngữ từng bình luận bằng AI để
tạo bộ dữ liệu ban đầu và chia train/validation/test, không chờ xác nhận từng
dòng. Nhãn AI phải có `label_source=ai`, `annotation_status=ai_labeled`;
không giả thành `human_confirmed`. Giữ nhãn người có xác nhận thật và dấu vết
đề xuất cũ. Câu chưa đủ căn cứ nằm trong unresolved và không vào bộ chia.
Các quy định dưới đây về chờ xác nhận người trước chia được áp dụng cho bộ
nhãn chuẩn do người gán; bộ AI này là ngoại lệ được người dùng cho phép.
Test mang nhãn AI chỉ đo mức phù hợp với bộ nhãn AI, chưa thay thế đánh giá
bằng nhãn con người độc lập. Giữ mọi yêu cầu về nguồn, đối soát và không rò rỉ.
Nhiệm vụ này không thu thập lại, không fit vectorizer/huấn luyện mô hình.

## 1. Mục tiêu và phạm vi

Hoàn thiện dự án `ptit-sentiment-analysis`, package Python `ptit_sentiment`.

Tên báo cáo: **Xây dựng hệ thống phân loại cảm xúc bình luận của sinh viên PTIT trên các nhóm Facebook**.

Mỗi bình luận nhận đúng một nhãn: `positive`, `neutral`, `negative`.

Sản phẩm: code chạy được, dữ liệu thực tế có nguồn gốc, nhãn được kiểm tra, mô hình được huấn luyện, đánh giá thực tế, chương trình dự đoán, báo cáo tiếng Việt và tài liệu giải thích. Không cần slide, giao diện web hoặc triển khai cloud.

Phải thực hiện đủ:
- BoW + Multinomial Naive Bayes.
- TF-IDF + Multinomial Naive Bayes.
- BoW + SVM tuyến tính.
- TF-IDF + SVM tuyến tính.
- Fine-tune PhoBERT cho cùng ba nhãn.
- Baseline dự đoán lớp phổ biến nhất trong train để đối chiếu.

## 2. Quy tắc làm việc

- Trước khi sửa, đọc hướng dẫn trong workspace, kiểm tra code, cấu hình, dữ liệu và trạng thái Git.
- Tái sử dụng cấu trúc hiện có; không tạo dự án song song hoặc ghi đè thay đổi của thành viên khác.
- Khi được yêu cầu triển khai, tạo/sửa file và chạy kiểm tra thực tế; không chỉ trả kế hoạch.
- Không để TODO, stub hoặc kết quả giả trong chức năng được báo là hoàn thành.
- Chủ động quyết định các chi tiết thông thường. Chỉ hỏi khi thiếu thông tin bắt buộc hoặc lựa chọn ảnh hưởng đáng kể đến phạm vi.
- Nếu bị chặn, hoàn thiện những phần độc lập còn lại và báo chính xác việc cần người dùng làm.
- Không tự publish, push, gửi dữ liệu ra dịch vụ ngoài hoặc thực hiện thao tác phá hủy ngoài phạm vi được yêu cầu.
- Không bịa dữ liệu thật, số liệu, nguồn tham khảo hoặc kết luận.
- Tra tài liệu chính thức khi cần xác minh API, phiên bản thư viện hoặc yêu cầu đầu vào PhoBERT.
- Viết code dễ đọc, thông báo lỗi hữu ích, docstring nêu đầu vào/đầu ra; tránh kiến trúc phức tạp không cần thiết.

## 3. Cấu trúc và môi trường

Nếu chưa có cấu trúc, dùng:

```text
configs/
data/raw/
data/labeled/
data/splits/
data/examples/
src/ptit_sentiment/
  labels.py
  data/
  preprocessing/
  features/
  models/
  training/
  evaluation/
  predict.py
artifacts/
outputs/
docs/
report/
pyproject.toml
README.md
.gitignore
```

- Có thể điều chỉnh theo code hiện có, giữ ranh giới trách nhiệm rõ ràng.
- Cài package để chạy được bằng `python -m`; không phụ thuộc thư mục làm việc ngẫu nhiên.
- Cấu hình đường dẫn, seed, tham số bằng YAML/JSON hoặc CLI; không hard-code đường dẫn máy cá nhân.
- Tách nhóm phụ thuộc mô hình truyền thống và PhoBERT; chỉ import thư viện nặng khi cần.
- README ưu tiên lệnh Windows PowerShell.
- Không commit credentials, cookies, dữ liệu có thông tin cá nhân, cache, checkpoint lớn hoặc output lớn. Cung cấp hướng dẫn tái tạo/tải artifact.

## 4. Dữ liệu thật và thu thập

- Kiểm tra dữ liệu thực có trước; ưu tiên file bình luận đã xuất hoặc nguồn được người dùng chỉ định và được phép truy cập.
- Nếu nguồn chưa có, hỏi danh sách URL hoặc file nhập; tiếp tục hoàn thiện luồng nhập, xử lý và huấn luyện trong lúc chờ.
- Không tự truy cập tài khoản Facebook hay khởi chạy thu thập khi chưa có nguồn và phương thức truy cập phù hợp.
- Khi được giao thu thập, triển khai giới hạn số lượng, phân trang theo nguồn, timeout, retry có giới hạn, checkpoint, tiếp tục lần chạy và chống trùng.
- Ghi lại số lượng thu thành công, nguồn lỗi và nguyên nhân; không vượt cơ chế kiểm soát truy cập.
- Nếu nguồn không hỗ trợ thu thập bằng cách hiện có, báo điểm bị chặn và cung cấp luồng nhập file xuất/thu thủ công.
- Mục tiêu ban đầu khoảng 2.000 bình luận hợp lệ từ nhiều bài PTIT; đây là mục tiêu, không phải số liệu đã đạt.
- Không lặp dữ liệu hoặc tạo câu giả để đạt số lượng; không thay bình luận PTIT bằng review sản phẩm và gọi đó là dữ liệu chính.
- Giữ dữ liệu gốc riêng, lưu nguồn và thời điểm thu trong metadata. Không yêu cầu lưu tên người viết hoặc thông tin nhận diện không cần thiết.

## 5. Hợp đồng dữ liệu và gán nhãn

CSV UTF-8 có `id`, `post_id`, `text`, `label`.

- Đọc `id`, `post_id` như chuỗi; `text` giữ nội dung gốc.
- Dữ liệu thô hoặc chờ gán nhãn được phép chưa có label. Dữ liệu huấn luyện phải có nhãn hợp lệ ở mọi dòng.
- Định nghĩa nhãn và ánh xạ số ở một nơi duy nhất; lưu ánh xạ trong artifact.
- `positive`: khen ngợi, hài lòng, ủng hộ; `negative`: bất mãn, chê trách, thất vọng; `neutral`: thông tin/hỏi đáp không có sắc thái rõ.
- Không coi câu khó xác định là neutral. Câu nhiều sắc thái chọn sắc thái chiếm ưu thế; câu mơ hồ đưa vào quy trình xem xét.
- Tạo hướng dẫn gán nhãn, file phân công, luồng ghép nhãn, kiểm tra nhãn chéo và xử lý bất đồng cho bốn người.
- AI hoặc mô hình chỉ được đề xuất `suggested_label`; không biến đề xuất thành nhãn chuẩn nếu chưa được người gán nhãn xác nhận.
- Không dùng mô hình để tạo nhãn test rồi lấy đó làm chuẩn đánh giá chính.
- Kiểm tra cột, dữ liệu rỗng, nhãn sai, id trùng, nội dung trùng và thống kê các mẫu bị loại.

## 6. Chia tập và tránh rò rỉ

- Dùng cùng bộ chia cho mọi mô hình. Mặc định train/validation/test 70/15/15, seed cố định.
- Ưu tiên chia theo `post_id`; bình luận cùng bài không được nằm ở nhiều tập. Nếu thiếu thông tin nhóm, ghi rõ hạn chế, không chế tạo post_id để giả vờ chia theo bài.
- Kiểm tra giao nhau theo id, post_id và nội dung chuẩn hóa dùng để phát hiện trùng. Xem xét các câu gần trùng.
- Cố gắng giữ phân bố nhãn gần nhau; xuất tỷ lệ thực tế. Mỗi tập phải có đủ ba nhãn.
- Nếu không thể chia hợp lệ, báo rõ và yêu cầu dữ liệu bổ sung; không âm thầm dùng cách chia gây rò rỉ.
- Lưu bộ chia, seed, cách chia và dấu vết phiên bản dữ liệu.
- Chỉ học vectorizer và các phép biến đổi có tham số từ train.
- Validation dùng chọn tham số/checkpoint. Test chỉ dùng đánh giá cuối cùng sau khi chốt cấu hình.
- Không điều chỉnh mô hình dựa trên test rồi vẫn gọi kết quả đó là đánh giá độc lập.

## 7. Tiền xử lý và đặc trưng

- Chuẩn hóa Unicode, khoảng trắng, ký tự lỗi, URL và tiếng lóng rõ nghĩa.
- Tách từ tiếng Việt; lựa chọn công cụ theo môi trường và tài liệu chính thức.
- Xử lý emoji, dấu câu, stop words có cấu hình và giải thích. Giữ phủ định như “không”, “chưa”, “chẳng” và dấu hiệu cảm xúc.
- Không ghi đè text gốc; cung cấp ví dụ trước/sau xử lý.
- BoW và TF-IDF hỗ trợ unigram/unigram–bigram, cấu hình `min_df`, `max_df`, `max_features` và tham số phù hợp.
- PhoBERT có đường chuẩn bị đầu vào riêng theo checkpoint; không áp dụng máy móc toàn bộ tiền xử lý truyền thống.
- Ghi chính sách truncation và thống kê mẫu bị cắt khi chạy PhoBERT.

## 8. Huấn luyện và lưu mô hình

### Mô hình truyền thống
- Chạy đủ bốn tổ hợp; dùng lưới nhỏ cho alpha của NB, C của SVM và đặc trưng.
- Chọn bằng Macro-F1 validation; baseline học lớp phổ biến nhất từ train.
- Lưu pipeline hoàn chỉnh gồm tiền xử lý, vectorizer và classifier bằng joblib; các thành phần phải tải lại được.
- Lưu cấu hình, seed, phiên bản thư viện, thông tin dữ liệu, log và kết quả validation.
- Mặc định giữ mô hình fit trên train và được chọn bằng validation để đánh giá test. Nếu dùng train+validation để refit, phải chốt tham số trước, ghi rõ chính sách và áp dụng nhất quán khi so sánh.

### PhoBERT
- Cài đặt fine-tune thật, không chỉ tải checkpoint pretrained.
- Cấu hình checkpoint, batch size, learning rate, số epoch, max length và thiết bị.
- Dùng cùng bộ chia và nhãn; chọn checkpoint bằng Macro-F1 validation.
- Lưu model, tokenizer, ánh xạ nhãn và cấu hình.
- Nếu có dữ liệu thật và tài nguyên phù hợp, thực hiện huấn luyện theo yêu cầu; nếu bị chặn, báo nguyên nhân và chuẩn bị notebook/lệnh GPU cùng hướng dẫn đưa artifact về.
- Không tự thay PhoBERT bằng mô hình khác hoặc coi pretrained model là mô hình đã fine-tune.
- Theo dõi đúng trạng thái: code đã viết, đã kiểm tra tối thiểu hay đã huấn luyện đầy đủ.

## 9. Đánh giá và dự đoán

- Đánh giá Accuracy; Precision, Recall, F1, support theo nhãn; Macro-Precision, Macro-Recall, Macro-F1.
- Confusion Matrix 3x3, thứ tự nhãn cố định; ghi rõ trục nhãn thật/dự đoán.
- Xuất `metrics.json`, `classification_report.csv`, `confusion_matrix.png`, `predictions.csv`, `errors.csv`.
- Dự đoán đánh giá có `id`, `text`, `true_label`, `predicted_label`, `model_name`.
- Kết quả lưu tên model, phiên bản dữ liệu/bộ chia và cấu hình để truy vết.
- Chỉ so sánh trực tiếp kết quả trên cùng phiên bản test; không trộn dữ liệu thật và minh họa.
- Phân tích lỗi bằng ví dụ thật đã loại thông tin nhận diện; tách quan sát và giả thuyết nguyên nhân.
- CLI hỗ trợ một bình luận hoặc CSV, chọn model đã lưu, xuất nhãn; không train lại lúc predict.
- Dữ liệu dự đoán mới không cần label. Lệnh đánh giá phải yêu cầu nhãn chuẩn.
- Báo lỗi rõ với đầu vào rỗng, file sai định dạng và artifact thiếu.
- Không coi decision score SVM là xác suất hay xác suất nhãn là độ mạnh cảm xúc.

## 10. Báo cáo và tài liệu

Viết tiếng Việt trong `report/report.md`: giới thiệu, lý thuyết, dữ liệu/gán nhãn, tiền xử lý, thiết kế thực nghiệm, kết quả, phân tích lỗi, hạn chế/kết luận và tài liệu tham khảo.

- Số liệu lấy từ output thực tế có dấu vết lần chạy; không điền kết quả dự kiến như kết quả đã đạt.
- Phần chưa thực nghiệm ghi rõ, không kết luận mô hình nào tốt nhất trước khi đánh giá.
- Chỉ đưa nguồn đã xác minh hoặc ghi rõ nguồn cần xác minh.
- Tạo `docs/task_assignment.md`, `docs/data_contract.md`, `docs/labeling_guide.md`, `docs/methodology.md`, `docs/learning_guide.md`.
- Tài liệu học giải thích luồng hệ thống, fit/transform/predict, BoW/TF-IDF, NB/SVM, pretrain/fine-tune, độ đo ba nhãn và Confusion Matrix; liên kết tới code tương ứng.
- README ghi lệnh thực sự chạy được cho validate, split, train, evaluate, compare, predict; cài PhoBERT riêng; dữ liệu cần có và cách tái lập.

## 11. Phân công bốn thành viên

| Người | Trách nhiệm | Đầu vào | Đầu ra |
|---|---|---|---|
| 1 | Thu thập, điều phối gán nhãn, kiểm tra/chia dữ liệu; ghép báo cáo | Nguồn bình luận thật, nhãn thành viên | Dữ liệu, hướng dẫn nhãn, bộ chia, thống kê, phần báo cáo dữ liệu |
| 2 | Tiền xử lý, BoW, TF-IDF | Bộ dữ liệu người 1 | Code tiền xử lý/đặc trưng, cấu hình, ví dụ, phần báo cáo tương ứng |
| 3 | NB, SVM, đánh giá, phân tích lỗi | Dữ liệu và phần người 2 | Bốn pipeline, code đánh giá, kết quả, phần báo cáo truyền thống |
| 4 | PhoBERT, tích hợp dự đoán, hướng dẫn chạy | Dữ liệu, pipeline người 3 | PhoBERT, CLI, tài liệu chạy, kết quả/phần báo cáo mô hình hiện đại |

Cả nhóm cùng gán nhãn và rà soát báo cáo. Mỗi người hiểu đầu vào, xử lý, đầu ra, lựa chọn và lỗi của phần mình. Người 3 và 4 phối hợp tổng hợp so sánh.

## 12. Kiểm chứng và định nghĩa hoàn thành

- Dữ liệu tổng hợp chỉ được dùng ở `data/examples/` để kiểm tra code, gắn nhãn rõ và lưu output riêng; không thay thực nghiệm thật.
- Kiểm tra có ý nghĩa: schema/nhãn, chia tập không rò rỉ, vectorizer chỉ fit train, train/evaluate bốn mô hình, lưu/tải giữ dự đoán, CLI một câu/CSV, định dạng kết quả và đối chiếu phiên bản test.
- Kiểm tra PhoBERT theo khả năng tài nguyên; ghi rõ phần chưa chạy. Không tạo test hình thức chỉ để tăng số lượng.
- Khắc phục lỗi đã phát hiện và chạy lại kiểm tra liên quan.
- Phân biệt code hoàn thiện, thu dữ liệu thật, xác nhận nhãn, train thật, đánh giá thật và cập nhật báo cáo.
- Chỉ tuyên bố hoàn thành thực nghiệm khi có dữ liệu thật, nhãn được kiểm tra, mô hình đã huấn luyện và đánh giá test thật. Nếu thiếu một mô hình, báo là chưa đủ yêu cầu.

Sau mỗi nhiệm vụ, báo ngắn gọn: thay đổi, kiểm chứng thực tế, phần chưa chạy/bị chặn và bước cụ thể tiếp theo. Khi bàn giao toàn bộ, thêm số lượng/phân bố dữ liệu thật, mô hình đã train, kết quả, lệnh sử dụng và đường dẫn báo cáo.
