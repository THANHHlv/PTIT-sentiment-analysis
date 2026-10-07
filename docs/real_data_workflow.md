# Vận hành với bình luận PTIT thật

> Cập nhật 07/10/2026: data đã thu gọn, chỉ còn labeled_comments.csv và
> train/validation/test cùng manifest.json chứa metadata nhúng. Hồ sơ gán
> nhãn, dữ liệu thô, unresolved/excluded đã lưu riêng trong
> outputs/archives/data_minimal_20261007.zip. Các đường dẫn hồ sơ/batch trong
> nội dung lịch sử dưới đây chỉ dùng sau khi khôi phục ZIP, không có trong
> data hiện tại. Kiểm tra bộ đang dùng bằng scripts/verify_ai_dataset.py;
> đóng gói bằng scripts/package_team_data.py. Fixture ở tests/fixtures/.

**Tài liệu lịch sử.** Ngày 07/10/2026, assignments_v1/confirmed_v1/real_v1
đã được bỏ khỏi data vì không có xác nhận người thật. Không chạy các lệnh
trỏ vào các bản đó. Luồng hiện tại ở [annotation_review.md](annotation_review.md),
dữ liệu sử dụng ở [data/README.md](../data/README.md). Bản khôi phục cũ nằm
trong outputs/archives/data_cleanup_20261007.zip.

## Trạng thái nguồn — cập nhật 03/10/2026 17:40

Hai URL người dùng cung cấp:

| Nhóm | URL | Kiểm tra thực tế |
|---|---|---|
| Cộng đồng sinh viên PTIT | https://www.facebook.com/groups/2k5ptit | HTTP 200, HTML công khai chỉ đọc được tiêu đề; không có link bài hoặc đối tượng bình luận trong lần kiểm tra |
| Góc Thông Tin PTIT | https://www.facebook.com/groups/584397217391365 | HTTP 200, cùng giới hạn trên |

Người dùng xác nhận xem được khi chưa đăng nhập. Công cụ trình duyệt phiên này không có browser kết nối;
khởi tạo browser trả `Browser is not available: iab`. Không kết luận nhóm riêng tư từ lỗi công cụ.
`outputs/real_access_v1/public_access.json` ghi timestamp, HTTP status và số đối tượng Comment tìm được
trong các script JSON của HTML ban đầu. Kiểm tra riêng văn bản và liên kết HTML cũng chỉ thấy tên nhóm,
không có link bài viết. Không có bình luận nào được thu trong lần kiểm tra này.

`probe_public_sources.py` là kiểm tra khả năng truy cập, **không phải crawler hoàn chỉnh**.
Không gọi API nội bộ, không lấy cookie, không vượt login/CAPTCHA. Chưa có adapter Facebook trực tiếp,
phân trang hoặc mã bài/bình luận được xác minh. Không ghi 2.000 là số đã thu.
Luồng đang hoạt động là nhập file xuất theo từng đợt; phân trang là các file người thu cung cấp,
không phải tự phân trang Facebook. Timeout/retry hiện chỉ áp dụng ở bước kiểm tra HTTP.

## Việc cần có để tiếp tục thu thực tế

Mở nguồn trong trình duyệt được kết nối với phiên công cụ để kiểm tra giao diện bài và bình luận,
hoặc xuất/ghi thủ công bình luận bạn được phép sử dụng thành CSV UTF-8.
Không gửi mật khẩu, cookie hoặc token trong chat. Không giả định Download Your Information của Facebook
xuất toàn bộ bình luận của các thành viên khác trong nhóm.

File nhập cần `id,post_id,text`. Dùng mã bình luận và mã bài thật từ liên kết/nguồn xuất;
không tạo post_id giả để vượt kiểm tra chia nhóm. Giữ nội dung nguyên văn và không cần tên tác giả.
Ghi chú quyền truy cập, URL nhóm và danh sách file trong `configs/collection.yaml`.
Các cột phụ của file xuất được bỏ khi nhập; bộ huấn luyện chỉ giữ schema tối thiểu.
Không cần tạo dòng ví dụ giả trong thư mục raw.

## Nhập từng đợt

Điền `permission_note` đúng thực tế và `exports` cho hai nguồn đã cấu hình. Đường dẫn file
được tính từ thư mục chạy lệnh, nên chạy tại gốc dự án. Dùng tên file mới cho mỗi đợt;
không sửa file đã nhập vì hash/checkpoint sẽ phát hiện thay đổi.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m ptit_sentiment.data.collect --config configs/collection.yaml --output-dir data/raw/ptit_sources_v1
```

- `collection.sqlite3`: dữ liệu và checkpoint transaction, tiếp tục cùng thư mục khi gián đoạn.
- `comments.csv`: snapshot lưu sau mỗi batch, chứa id/post_id/text và nguồn/thời điểm nhập.
- `collection_report.json`: hash file, số id/bài, lỗi từng nguồn, dòng loại, id trùng, nhóm nội dung trùng.
- Mặc định dừng khi đạt 2.000 id hợp lệ. Chưa đạt vẫn báo số thực; không lặp dữ liệu cho đủ.
- Cùng id/cùng nội dung được bỏ trùng. Cùng id khác bài/nội dung bị cách ly trong danh sách lỗi.
- Nội dung giống nhau khác id được liệt kê; không tự quyết định giữ dòng nào. Hãy kiểm tra nguồn,
  sửa bộ file nhập thành một phiên bản đã rà soát và nhập vào thư mục mới trước khi gán nhãn.
- Lỗi một nguồn giữ phần đã nhập nhưng trả exit code khác 0. Không lấy báo cáo nguồn lỗi làm chứng cứ hoàn thành.
- Metadata nguồn là lời khai người thu và dấu vết file, không tự chứng minh nguồn là Facebook thật.

## Gán nhãn và kiểm tra chéo

```powershell
& $python -m ptit_sentiment.data.labeling prepare --input data/raw/ptit_sources_v1/comments.csv --source-report data/raw/ptit_sources_v1/collection_report.json --output-dir data/labeled/assignments_v1 --provenance real --cross-fraction 0.2 --seed 42
```

Bốn bảng `annotator_1.csv` đến `annotator_4.csv` phân công mỗi câu đúng một người chính;
20% câu có một người khác kiểm tra độc lập. `base.csv` giữ nguyên đầu vào và manifest ghi mọi phân công.
Không sửa id/post_id/text/role. Điền `label` và chỉ điền `confirmed=yes` sau khi người đọc xác nhận;
`notes` ghi giải thích/mơ hồ. Giữ id dưới dạng chuỗi khi mở bằng bảng tính để tránh mất số 0 đầu mã.

Tôi có thể đọc nội dung thật rồi gợi ý `suggested_label` theo yêu cầu, nhưng chưa có nội dung để gợi ý.
Nhãn gợi ý không được tự chép vào gold. Người kiểm tra chéo nên gán độc lập, không xem gợi ý trước.
Hướng dẫn mỉa mai, phủ định và thiếu ngữ cảnh: `docs/labeling_guide.md`.

```powershell
& $python -m ptit_sentiment.data.labeling merge --assignment-dir data/labeled/assignments_v1 --output-dir data/labeled/merge_review_v1
```

Nếu còn nhãn thiếu/sai/chưa xác nhận hoặc bất đồng, chưa xuất `labeled.csv`.
Sửa nhãn còn thiếu trong bảng thành viên. Để phân xử bất đồng đã có hai nhãn xác nhận,
copy `needs_review.csv` thành file resolutions, điền `final_label,confirmed,reviewer,reason`;
`issue_reason` là lý do kỹ thuật, khác với `reason` giải thích quyết định của người phân xử.

```powershell
& $python -m ptit_sentiment.data.labeling merge --assignment-dir data/labeled/assignments_v1 --resolutions data/labeled/resolutions_v1.csv --output-dir data/labeled/confirmed_v1
```

Mỗi lần merge dùng thư mục mới, không ghi đè bản rà soát. Mặc định yêu cầu mọi câu xác nhận,
đủ kiểm tra chéo và tỷ lệ đồng thuận ban đầu >= 0,8. Đồng thuận thấp cần xem lại quy tắc và gán lại,
không giảm ngưỡng để làm đẹp dữ liệu. Kappa gộp các cặp người khác nhau chỉ là chẩn đoán mô tả,
không phải hệ số nhiều người gán nhãn. Lưu cả báo cáo chất lượng và các lần phân xử.
Xác nhận trong CSV là dấu vết khai báo của người gán, không phải chữ ký số xác thực danh tính.

## Chia tập thật và huấn luyện

```powershell
& $python -m ptit_sentiment.data.validate --input data/labeled/confirmed_v1/labeled.csv
& $python -m ptit_sentiment.data.split --input data/labeled/confirmed_v1/labeled.csv --labeling-report data/labeled/confirmed_v1/labeling_report.json --output-dir data/splits/real_v1 --provenance real --config configs/classical.yaml
& $python scripts/run_real.py train --splits-dir data/splits/real_v1 --artifact-dir artifacts/real_v1 --output-dir outputs/real_v1 --skip-phobert
```

Lệnh train này chạy bốn classical và baseline. Muốn đủ PhoBERT, sau khi cài phụ thuộc, Java17,
RDRSegmenter và checkpoint như README, chạy riêng:

```powershell
$env:HF_HOME = Join-Path (Get-Location) ".cache/huggingface"
& $python -m ptit_sentiment.training.train_phobert --splits-dir data/splits/real_v1 --config configs/phobert_gpu4gb.yaml --artifact-dir artifacts/real_v1/phobert
```

Hoặc bỏ `--skip-phobert` ngay từ lệnh run_real train để chạy cả hai phần trên thư mục artifact mới.
Cấu hình GPU 4 GB dùng batch1, accumulation8, gradient checkpointing và fp16, 3 epoch, chọn bằng validation.
Đây là cấu hình thực nghiệm dự kiến; **chưa chạy bằng dữ liệu thật**, không bảo đảm bộ nhớ trên mọi máy.
Với CPU đặt `use_cpu: true`, `fp16: false` và `local_files_only: false` nếu cần tải checkpoint.
Không thay PhoBERT bằng mô hình khác khi thất bại. Lệnh không tự sửa dữ liệu hay bộ chia.

## Đánh giá độc lập sau khi chốt cấu hình

```powershell
& $python scripts/run_real.py evaluate --splits-dir data/splits/real_v1 --artifact-dir artifacts/real_v1 --output-dir outputs/real_v1
& $python -m ptit_sentiment.predict --model artifacts/real_v1/classical/tfidf_svm.joblib --text "Mình chưa hài lòng với lịch học."
& $python -m ptit_sentiment.predict --model artifacts/real_v1/classical/tfidf_svm.joblib --input data/raw/new_comments.csv --output outputs/new_predictions.csv
```

Nếu chỉ đã train classical, thêm `--skip-phobert` khi evaluate và ghi rõ chưa đủ sáu mô hình.
Evaluate không train lại; xuất metrics, báo cáo từng nhãn, confusion matrix, predictions và errors cho
mỗi mô hình, rồi tổng hợp bảng/biểu đồ cùng phiên bản test. Đọc errors sau đánh giá để phân tích;
không điều chỉnh mô hình theo test rồi gọi lần đánh giá đó là độc lập.
Báo cáo cần lấy số từ `outputs/real_v1/evaluation/*/metrics.json` và `comparison.csv`.
Chưa có các output thật này thì phần kết quả thật giữ trạng thái chưa thực nghiệm.

## Kiểm chứng code

```powershell
& $python -m unittest discover -s tests -v
```

22 kiểm tra đã qua, bao gồm resume/cap, hash thay đổi, id chuỗi, dòng trùng/xung đột, schema sai,
nhãn gợi ý không thành gold, bất đồng/phân xử, bất biến text và xác nhận nhãn/hash của bộ chia thật.
Fixture chỉ nằm trong thư mục tạm, không phải dữ liệu Facebook hay bằng chứng thực nghiệm thật.
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
