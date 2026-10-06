# Gán nhãn AI ban đầu và kiểm tra dữ liệu — 06/10/2026

> Cập nhật 07/10/2026: data đã thu gọn, chỉ còn labeled_comments.csv và
> train/validation/test cùng manifest.json chứa metadata nhúng. Hồ sơ gán
> nhãn, dữ liệu thô, unresolved/excluded đã lưu riêng trong
> outputs/archives/data_minimal_20261007.zip. Các đường dẫn hồ sơ/batch trong
> nội dung lịch sử dưới đây chỉ dùng sau khi khôi phục ZIP, không có trong
> data hiện tại. Kiểm tra bộ đang dùng bằng scripts/verify_ai_dataset.py;
> đóng gói bằng scripts/package_team_data.py. Fixture ở tests/fixtures/.

Bước gán nhãn AI và chia dữ liệu đã hoàn tất trong phạm vi hiện tại. Không
thu thập lại, không gọi API/dịch vụ ngoài, không fit vectorizer hoặc train.
Người dùng cho phép nhãn AI làm bộ dữ liệu ban đầu. **Chưa có nhãn con người
được xác nhận thật trong đợt này.** Test mang nhãn AI chỉ đo mức phù hợp với
bộ nhãn AI, chưa thay thế đánh giá bằng nhãn con người độc lập.

## Đầu vào và dấu vết phiên bản

Bản chính: `data/raw/ptit_sources_v1/comments.csv`, UTF-8,
`id,post_id,text,source_id,source_url,collected_at`. Đọc mọi mã như chuỗi,
giữ nguyên text, post_id và metadata nguồn. Không tạo post_id giả.
SHA256: `5bf42ef6361a7e7c7cefe1e60aa04fd91e52ed7c91df3a82e1c048cf28b31287`.

| Nguồn metadata | Bình luận trong bản chính |
|---|---:|
| ptit_2k5 | 2.377 |
| ptit_group_584397217391365 | 487 |
| Tổng — 453 bài viết | 2.864 |

Database collection.sqlite3 có cùng tập id/post_id/text, đã kiểm tra chỉ
đọc. 466 JSON trong browser_capture gồm 465 snapshot có bình luận và một
progress.json không có dòng. Tổng snapshot 3.001 dòng; 2.864 dòng khớp bản
chính, 137 ID khác đã được importer cũ loại do trùng nội dung. Không ghép
lại. `duplicate_occurrences.csv` giữ 137 quan hệ ID đại diện/bài/nguồn;
không suy ra đó là cùng một người. Ba file incoming có 47 dòng: 46 ID đã
có và một nội dung trùng, không có dữ liệu mới.

Nhãn cũ trong assignments_v1/confirmed_v1 chưa được người xác nhận theo trả
lời của người dùng; script cũ điền xác nhận và tạo bất đồng giả. Ngày 07/10/2026
đã bỏ các file cũ và bộ chia real_v1 khỏi data; bản khôi phục nằm trong
`outputs/archives/data_cleanup_20261007.zip`. Không dùng làm nhãn chuẩn hoặc
bộ chia hiện tại. Các đề xuất mới được đọc trực tiếp từng câu và lưu riêng.

## Kết quả gán nhãn và đối soát

| Chỉ tiêu | Số lượng |
|---|---:|
| Đầu vào | 2.864 |
| Qua kiểm tra cấu trúc, đưa vào batch | 2.856 |
| Đã đọc và lưu quyết định, 18 batch | 2.856 |
| Chưa xử lý | 0 |
| Còn hợp lệ sau lọc chất lượng | 2.794 |
| Có nhãn AI | 2.600 |
| Chưa đủ căn cứ — unresolved | 194 |
| Bị loại | 70 |
| Trong đó lỗi cấu trúc | 8 |
| Trong đó loại sau đọc trực tiếp | 62 |
| Thiếu id/post_id/text hoặc metadata nguồn | 0 |
| Trùng ID / nội dung chuẩn hóa trong bản chính | 0 / 0 |
| Cặp gần trùng đã nhận diện | 13 |
| Có nhãn AI nhưng ưu tiên kiểm tra high | 487 |
| Nhãn người xác nhận thật | 0 |

Đối soát: **2.600 + 194 + 70 = 2.864**, mỗi ID đúng một đích, không mất
hoặc nhân bản. 137 bản trùng đã bị bỏ trước khi xuất bản chính được ghi
quan hệ riêng, không cộng vào mẫu đầu vào lần nữa.

| Nhãn | AI — ai_labeled | Người — human_confirmed |
|---|---:|---:|
| positive | 362 | 0 |
| neutral | 1.800 | 0 |
| negative | 438 | 0 |

Không sửa nhãn để cân bằng lớp. Mỗi nhãn AI có reason và review_priority;
ưu tiên không phải xác suất. Phân tích toàn câu, phủ định, đối lập, tiếng
lóng và mỉa mai theo [labeling_guide.md](labeling_guide.md); không dùng luật
từ khóa hoặc mô hình để điền nhãn. Không suy nhãn neutral khi không hiểu.
Chỉ dùng text thật của bình luận; dữ liệu hiện không có text bài/cha.

unresolved gồm phản ứng/tiếng cười thiếu ngữ cảnh, từ lóng chưa rõ nghĩa,
câu bỏ lửng, tham chiếu không rõ, đa sắc thái chưa có phần trội. Giữ text
và reason từng dòng; không loại vì khó. Các dấu hiệu ba chấm, giao diện,
quảng bá là cờ kiểm tra, không phải luật tự xóa. Một số dấu ba chấm có thể
là chủ ý; snapshot không đủ chứng minh mọi nội dung đã thu đầy đủ.

## Bộ chia thực tế

Seed **42**, mục tiêu **70/15/15**, 10.000 ứng viên chia nhóm. Đơn vị chia
là post_id thật; nối các bài qua 13 cặp gần trùng và 137 quan hệ trùng lịch
sử trước khi chia. 446 bài còn nhãn hợp lệ tạo 370 thành phần nhóm chia.
`split_component` là mã nhóm nội bộ riêng, không thay post_id gốc.

| Tập | Dòng | Bài | positive | neutral | negative | Tỷ lệ dòng |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,801 | 326 | 254 | 1246 | 301 | 69.27% |
| validation | 409 | 63 | 58 | 283 | 68 | 15.73% |
| test | 390 | 57 | 50 | 271 | 69 | 15.00% |

Cả ba tập đủ ba nhãn. Giao nhau theo ID, post_id, nội dung NFC/casefold/gộp
khoảng trắng, cặp gần trùng đã nhận diện và thành phần nhóm đều bằng **0**.
Tìm gần trùng dùng Jaccard từ >=0,6 và tỷ lệ ký tự >=0,9 trên câu từ 12 ký
tự; chưa bảo đảm phát hiện mọi diễn đạt lại cùng nghĩa. Giữ nguyên mọi metadata
nhãn/nguồn trong train/validation/test; không đưa unresolved vào các tập.

Phiên bản: `ai_initial_4e93595ef7d1_seed42`. Hash dữ liệu:
`4e93595ef7d19e1b60d71e44f83cdeabc2790ef15b0113831b0e59ee2a331dc4`. Metadata lưu trong `data/splits/manifest.json`
và `metadata.json`. Ngày 07/10/2026 đã bỏ thư mục versions vì cả tám file
trùng byte với bộ hiện tại; phiên bản và hash vẫn giữ trong manifest.
Bộ chia cũ được chuyển vào ZIP khôi phục, chỉ giữ bộ tổng hợp complete_20261003
đang được artifact/báo cáo minh họa tham chiếu. Công cụ từ chối ghi đè bộ chia.

## Các file cần mở

- `data/labeled/labeled_comments.csv`: 2.600 dòng có label, label_source=ai,
  annotation_status=ai_labeled, reason, review_priority và metadata nguồn.
- `data/labeled/unresolved.csv`: 194 dòng label trống, trạng thái unresolved,
  lý do chưa xác định; không vào bộ chia.
- `data/labeled/excluded_comments.csv`: 70 dòng, lý do và giai đoạn loại.
- `data/labeled/comments_clean.csv`: 2.794 dòng giữ text gốc.
- Toàn bộ 2.794 quyết định giữ lại nằm trong labeled_comments + unresolved;
  đã bỏ ai_annotations.csv vì là bản tổng hợp trùng dữ liệu.
- `data/labeled/label_suggestions.csv`: giữ dấu vết đề xuất; label trống,
  pending_review, không phải file nhãn AI cuối cùng.
- `data/labeled/review_sample.csv`: **771 ID riêng**, gồm 681 high (194
  unresolved và 487 có nhãn) cùng 90 mẫu ngẫu nhiên low/medium, 30 mỗi lớp,
  seed 42. sample_role phân biệt nhóm; kiểm tra cả câu bình thường.
- `data/labeled/review_required.csv`: bảng đề xuất cho 681 dòng high.
- `data/labeled/id_reconciliation.csv`: đích labeled/unresolved/excluded của
  mọi ID; duplicate_occurrences.csv/near_duplicates.csv lưu quan hệ trùng.
- `data/splits/train.csv`, `validation.csv`, `test.csv`: bộ chia mới.
- `outputs/data_audit.json`, `outputs/ai_dataset_verification.json`: số liệu,
  nguồn, hash và kiểm chứng thực tế; không chứa nguyên văn bình luận.

Dữ liệu riêng có thể chứa tag/liên hệ trong text gốc; được .gitignore loại
khỏi Git, không commit hoặc chia sẻ công khai. Không xuất danh tính người
viết trong tài liệu/log.

CSV UTF-8 BOM. Trên Windows, dùng **Data → From Text/CSV → Transform Data**,
chọn UTF-8/comma, bỏ Changed Type và đặt **mọi cột thành Text** trước Load.
Không nhấp đúp CSV: mã dài/số 0 đầu có thể đổi, và nội dung bắt đầu =,+,-,@
cần được nhập dạng văn bản. CSV giữ text gốc, không chèn ký tự làm đổi câu.

## Kiểm tra độc lập và xác nhận người — tùy chọn tiếp theo

Đợt mới `data/labeled/annotation_batches_ai_v1` có 2.794 ID, mỗi ID một
primary, 40 mẫu chung cho cả bốn, high và mẫu ngẫu nhiên có cross_check.
Không hiển thị gợi ý AI trong file thành viên để giảm ảnh hưởng đề xuất.
Phân công cũ giữ nguyên, không dùng tiếp vì danh sách sạch đã thay đổi.

| Người | File trong annotation_batches_ai_v1 | Lượt đọc | reviewer |
|---|---|---:|---|
| 1 | annotator_1.csv | 921 | member_1 |
| 2 | annotator_2.csv | 913 | member_2 |
| 3 | annotator_3.csv | 914 | member_3 |
| 4 | annotator_4.csv | 924 | member_4 |

Đọc độc lập, điền/sửa label thành positive/neutral/negative. Chỉ sau khi
chính người duyệt đọc và xác nhận mới đặt annotation_status=human_confirmed,
reviewer=member_1..member_4, context_used=comment_only hoặc nguồn ngữ cảnh
thật, notes cho câu khó. Giữ nguyên id/post_id/text/role và mọi dòng. Câu
chưa rõ giữ label trống/pending_review. Sao chép suggested_label không được
coi là xác nhận. Không sửa file bộ chia hiện tại khi nhãn thay đổi; tạo phiên
bản mới để mọi mô hình dùng cùng một bộ chia.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m ptit_sentiment.data.annotation_review merge `
  --assignment-dir data/labeled/annotation_batches_ai_v1 `
  --output-dir data/labeled/human_review_ai_v1
```

Công cụ kiểm tra nhãn hợp lệ, ID tồn tại/trùng/thiếu, text/post_id/role,
reviewer và xác nhận tường minh. Xuất validation_issues.csv, needs_review.csv,
disagreements.csv. Chỉ các ID đủ người xác nhận và đồng thuận mới vào tập
người; không lấy đa số để tự kết luận. Có bất đồng thì copy disagreements,
thảo luận và điền final_label, annotation_status=human_confirmed, reviewer,
reason, rồi ghép với --resolutions và output mới. Không tạo kappa khi chưa
có nhãn độc lập thật. Merge giữ bộ AI hiện tại, xuất kết quả người riêng.

## Lệnh tái tạo và kiểm tra

Checkpoint `audit_base.csv` bất biến và `suggestion_batches/batch_*.json`
đã đủ mọi ID; mỗi ID đúng một quyết định. suggestion_progress.json ghi
remaining_rows=0. Hash ngăn resume nhầm phiên bản; ID đã xử lý bị từ chối.
Không chạy lại next_language_batch.json vì đó là batch đã lưu.

Tái xuất từ các quyết định đã đọc, giữ bộ chia hiện tại và chọn thư mục mới:

```powershell
& $python -m ptit_sentiment.data.ai_dataset `
  --directory data/labeled --output-dir data/splits/ai_initial_v2 `
  --seed 42 --attempts 10000
& $python scripts/verify_ai_dataset.py
& $python -m ptit_sentiment.data.validate --input data/labeled/labeled_comments.csv
& $python -m unittest discover -s tests -p test_ai_dataset.py -v
& $python -m unittest discover -s tests -p test_annotation_review.py -v
```

verify mặc định kiểm tra bộ chia gốc data/splits; dùng --split-dir nếu chọn
phiên bản khác. Kiểm chứng đã chạy: 5 test nguồn AI/giữ nhãn người/không chốt
checkpoint thiếu/đối soát roundtrip/liên kết nhóm bắc cầu, cùng 14 test schema,
resume, xác nhận/bất đồng/nhãn sai/mất dòng và chia tập. Tất cả 19 test đạt.
Kiểm tra dữ liệu thật đạt: raw/466 JSON không đổi, SQLite khớp, toàn bộ
id/text/post_id/nguồn giữ nguyên, đủ nhãn hợp lệ, metadata AI đúng, đối soát
mọi ID và không giao nhau giữa các tập. Không huấn luyện trong nhiệm vụ này.
