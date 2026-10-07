# Thư mục bàn giao có chọn lọc

> Cập nhật 07/10/2026: data đã thu gọn, chỉ còn labeled_comments.csv và
> train/validation/test cùng manifest.json chứa metadata nhúng. Hồ sơ gán
> nhãn, dữ liệu thô, unresolved/excluded đã lưu riêng trong
> outputs/archives/data_minimal_20261007.zip. Các đường dẫn hồ sơ/batch trong
> nội dung lịch sử dưới đây chỉ dùng sau khi khôi phục ZIP, không có trong
> data hiện tại. Kiểm tra bộ đang dùng bằng scripts/verify_ai_dataset.py;
> đóng gói bằng scripts/package_team_data.py. Fixture ở tests/fixtures/.

Bản đã tạo local: `handoff/PTIT_BAN_GIAO_20261007/`.
Chỉ đưa **nội dung `01_CODE_GITHUB/`** lên gốc kho GitHub, giữ cấu trúc
src/configs/docs/tests/scripts. Không upload cả thư mục bàn giao.

```text
PTIT_BAN_GIAO_20261007/
  README_BAN_GIAO.md
  MANIFEST_BAN_GIAO.json
  01_CODE_GITHUB/             code, cấu hình, tests, docs, ví dụ tổng hợp
  02_DATA_PRIVATE/
    TEAM_DATA.json
    data/labeled/            nhãn AI, unresolved, excluded, review và phân công người
    data/splits/             train, validation, test và metadata kiểm chứng
  03_HUONG_DAN_THANH_VIEN/    bốn hướng dẫn riêng và hướng dẫn merge nhãn người
```

Đây là bản sao bàn giao của dự án hiện có, không phải dự án mới để làm song
song. Tiếp tục phát triển trong cùng Git repository/package. Raw, browser
capture, dữ liệu cũ, script giả xác nhận, báo cáo cũ, cache, môi trường và
checkpoint không nằm trong bản bàn giao.

## Bộ dữ liệu

Version `ai_initial_4e93595ef7d1_seed42`: 2.600 nhãn AI; positive 362,
neutral 1.800, negative 438. Train/validation/test 1.801/409/390.
194 unresolved và 70 excluded không vào bộ chia. Chưa có nhãn người xác
nhận; giữ metadata nguồn AI, không gọi test là đánh giá nhãn người độc lập.

`02_DATA_PRIVATE/` giữ text gốc và nguồn để đối soát, chưa ẩn danh để công
khai. Chỉ chia sẻ nội bộ cho bốn thành viên được phép sử dụng. Mỗi thành
viên clone code, cài package, sao chép data/ và TEAM_DATA.json từ phần này
vào gốc clone; dùng cùng bộ chia, không tự chia lại. README code có lệnh
validate, train, evaluate, compare, predict và PhoBERT riêng.

## Phân công

| Người | Phần code | Đầu vào dùng chung |
|---|---|---|
| 1 | data/, hướng dẫn nhãn, split, điều phối báo cáo | labeled/unresolved, manifest, phân công rà nhãn |
| 2 | preprocessing/, features/ | train và cấu hình; fit vocabulary/IDF chỉ từ train |
| 3 | classical, train_classical, evaluation/ | bộ chia và phần tiền xử lý người 2 |
| 4 | PhoBERT, train_phobert, predict | cùng bộ chia, checkpoint và tài nguyên riêng |

Cả bốn cùng rà nhãn độc lập theo annotator_1..4.csv trong
`annotation_batches_ai_v1`. Nhãn đang trống/pending, không chứa gợi ý AI
trong file thành viên. Người 1 nhận file về và merge; output pending chưa
được biến thành human_confirmed. Bộ nhãn người mới phải phát hành version
mới, không sửa bộ AI đã khóa.

Bản sao assignment_manifest chỉ đổi clean_file thành đường dẫn tương đối
`data/labeled/comments_clean.csv`; CSV gốc và manifest split không đổi.
TEAM_DATA.json lưu hash bản nguồn và ghi rõ thay đổi cấu hình này.

## Tái tạo một bản bàn giao mới

```powershell
.\.venv\Scripts\python.exe scripts/prepare_team_handoff.py --output handoff/PTIT_BAN_GIAO_20261007_v2
```

Thư mục đích phải chưa có; công cụ từ chối ghi đè. Cần máy người 1 có dữ
liệu hiện hành và phân công. Công cụ chỉ sao chép/kiểm tra, không thu thập,
fit vectorizer hoặc train. README mẫu hiện mô tả version nêu trên;
khi phát hành dữ liệu version khác cần cập nhật nội dung mẫu trước đóng gói.
Sau khi nhận dữ liệu, các thành viên dùng package chung, không cần script
thu thập Chrome/CDP. Kiểm toán raw bằng verify_ai_dataset.py cần máy người
1 giữ đủ raw/snapshot/checkpoint.

Xem thêm [chính sách GitHub và dữ liệu](github_handoff.md) trong workspace
gốc. Bản code chọn lọc có hướng dẫn tương ứng tại README.md; thư mục
01_CODE_GITHUB không kèm file github_handoff.md lịch sử đó.
