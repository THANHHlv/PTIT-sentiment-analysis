# Kiểm chứng bàn giao — 03/10/2026

> Cập nhật 07/10/2026: data đã thu gọn, chỉ còn labeled_comments.csv và
> train/validation/test cùng manifest.json chứa metadata nhúng. Hồ sơ gán
> nhãn, dữ liệu thô, unresolved/excluded đã lưu riêng trong
> outputs/archives/data_minimal_20261007.zip. Các đường dẫn hồ sơ/batch trong
> nội dung lịch sử dưới đây chỉ dùng sau khi khôi phục ZIP, không có trong
> data hiện tại. Kiểm tra bộ đang dùng bằng scripts/verify_ai_dataset.py;
> đóng gói bằng scripts/package_team_data.py. Fixture ở tests/fixtures/.

## Kiểm tra trước sửa

Không tìm thấy AGENTS.md trong workspace hoặc các thư mục cha liên quan.
Git sạch tại commit abff874 (first commit); đã hoàn thiện trên cấu trúc hiện có.
data/labeled chỉ có template header, chưa có bình luận PTIT thật.
Không có dữ liệu/tài khoản gửi ra dịch vụ ngoài; chỉ tải thư viện và tài nguyên mô hình công khai.

## Luồng truyền thống

Lần complete_20261003 dùng data/examples/synthetic.csv: 60 câu tổng hợp, 20 bài giả định.
Train/validation/test: 42/9/9 câu; 14/3/3 post_id; mỗi tập đủ ba nhãn.
Đã chạy validate → split → 4 pipeline + baseline → lưu/tải → evaluate → compare → predict câu/CSV.
Không giao id/post_id/khóa text. Kiểm tra vocabulary thuộc đặc trưng train,
text gốc và nhãn được giữ đúng; SVM không có probability.
Đối chiếu classification_report.csv với metrics.json, kích thước/support/confusion matrix,
bảng so sánh 5 mô hình và PNG hợp lệ.
15 unittest đạt; compileall và git diff --check không báo lỗi.
Đã xem biểu đồ so sánh: chữ/trục/nhãn rõ ràng.

Output: outputs/complete_20261003, artifact: artifacts/complete_20261003.
commands.json và verification.json ghi vết chạy.
report/evidence lưu bản tóm tắt nhỏ, hash nguồn metric và biểu đồ để truy vết báo cáo.
Các điểm số là trên dữ liệu minh họa, không chứng minh chất lượng trên PTIT thật.

## PhoBERT thật

Đã cài transformers 4.44.2 và accelerate 0.34.2 vào .venv;
tái dùng torch 2.6.0+cu124 và py-vncorenlp 0.1.4.
Tải VnCoreNLP word segmenter và checkpoint vinai/phobert-base vào workspace,
checkpoint commit 01daacda68afe13d83023d16ec647239e344a1e6.
Tải qua mạng cần quyền ngoài sandbox; không tải lên dữ liệu bình luận.

Khởi tạo JVM Java 17 và RDRSegmenter thật thành công sau khi đặt JAVA_HOME.
Tắt TensorFlow integration để tránh xung đột Keras trong môi trường có sẵn.
Fine-tune giới hạn 1 bước trên RTX 3050 Laptop 4 GB, batch 1, max_length 64,
gradient checkpointing; không thay checkpoint bằng mô hình khác.

Lần chính: artifacts/phobert_verified_20261003.
Đã chọn/lưu checkpoint theo validation Macro-F1, lưu tokenizer/config/label map.
scripts/run_phobert_smoke.py --skip-train kiểm tra artifact:
- Nhãn validation tải lại khớp toàn bộ validation_reference.json từ mô hình trong bộ nhớ khi train.
- Evaluate validation và test dùng cùng bộ chia classical, xuất metric/CSV/PNG/errors.
- Predict một câu và CSV 3 câu chạy thành công; xác suất softmax và thống kê truncation được xuất.
- Train/validation/test có 0 mẫu bị cắt ở max_length 64 trên bộ minh họa này.
- verification.json báo passed; commands.json lưu exit code/stdout/stderr của các CLI.

Kết quả: outputs/phobert_verified_20261003; bằng chứng nhỏ: report/evidence/phobert_smoke.json.
Preflight chỉ kiểm tra sự hiện diện tài nguyên, không tự chứng nhận train:
trường runtime_training_verified=false của preflight có nghĩa “preflight không chạy train”;
chứng cứ runtime thực tế nằm ở verification.json của smoke.

## Môi trường và giới hạn

Python 3.12.9, sklearn 1.5.2, pandas 2.2.3, numpy 1.26.4, Underthesea 9.5.0,
joblib 1.4.2, PyYAML 6.0.2, matplotlib 3.9.2. .venv dùng --system-site-packages.
Chưa thử cài mới toàn bộ từ môi trường sạch; optional dependencies đã cài và runtime đã kiểm chứng.
Chưa chạy đầy đủ PhoBERT theo cấu hình thực nghiệm hoặc bất kỳ mô hình nào trên bình luận thật.
Cần dữ liệu thật, gán nhãn chéo, bộ chia khóa và thực nghiệm đầy đủ trước khi kết luận hiệu quả.

Các lần smoke cũ được giữ nguyên; không tự commit artifact, dữ liệu nhạy cảm, cache hoặc output lớn.
Báo cáo chính: report/report.md; tài liệu học phần 3: docs/learning_guide.md.

## Cập nhật nguồn thật và quy trình nhãn (03/10/2026)

Đã nhận hai URL nhóm PTIT và kiểm tra công khai HTTP 200, đúng tên nhóm; HTML đọc được chỉ có tiêu đề, không có link bài/bình luận.
Không có browser kết nối nên chưa kiểm tra feed JavaScript. Bình luận thật đã thu: 0.
Nhập file thực tế bằng collect hiện báo chưa có exports, giữ báo cáo 0 dòng.
22 unittest qua, gồm 7 kiểm tra mới về resume/cap/checkpoint, trùng/xung đột/schema,
nhãn gợi ý không thành gold, nhãn xác nhận/phân xử/bất biến text và audit bộ chia thật.
Các fixture tạm không phải dữ liệu thật; chưa vận hành train/evaluate thật.
Cấu hình PhoBERT GPU 4 GB 3 epoch chưa kiểm chứng đầy đủ. Luồng mới xem real_data_workflow.md.

## Kiểm tra tiếp trên bài viết cụ thể

Đã kiểm tra trực tiếp Facebook các bài `919830310226147`, `1064002089142301`,
`1049953217213855` thuộc nhóm `2k5ptit`. URL ứng viên tìm được qua chỉ mục công khai,
lưu trong `configs/public_posts.yaml`; không dùng nội dung trang chỉ mục làm dữ liệu huấn luyện.
Cả ba HTTP 200 với tiêu đề khớp bài viết, nhưng văn bản HTML chỉ có một phần tiêu đề,
không có link bài khác hay đối tượng Comment từ JSON chuẩn/JSON bootstrap đọc được.
Kết quả có timestamp tại `outputs/real_access_v1/public_posts_access.json`.
Đây là các URL bài đã kiểm tra, không phải ba bình luận thu được. Tổng bình luận thật vẫn 0.

Đã thử thêm backend Windows của skill Computer Use, lỗi native pipe không tồn tại.
Chưa thể điều khiển trình duyệt JavaScript ở máy từ phiên này. Cần kết nối trình duyệt/backend
hoặc cung cấp file xuất thật để tiếp tục; không thể khắc phục bằng thêm lần gọi HTTP giống nhau.
