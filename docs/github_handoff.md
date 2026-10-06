# GitHub và bàn giao sau phần 1

> Cập nhật 07/10/2026: data đã thu gọn, chỉ còn labeled_comments.csv và
> train/validation/test cùng manifest.json chứa metadata nhúng. Hồ sơ gán
> nhãn, dữ liệu thô, unresolved/excluded đã lưu riêng trong
> outputs/archives/data_minimal_20261007.zip. Các đường dẫn hồ sơ/batch trong
> nội dung lịch sử dưới đây chỉ dùng sau khi khôi phục ZIP, không có trong
> data hiện tại. Kiểm tra bộ đang dùng bằng scripts/verify_ai_dataset.py;
> đóng gói bằng scripts/package_team_data.py. Fixture ở tests/fixtures/.

## Quyết định cho dự án này

Đưa code, cấu hình tái lập, tài liệu và dữ liệu minh họa lên GitHub.
Dữ liệu Facebook thật hiện tại chia sẻ qua một gói riêng có quyền truy cập
cho bốn thành viên. Mọi người clone code, nhận đúng gói rồi làm trên cùng
bộ chia; không cần thu thập lại hoặc tự chia trên từng máy.

Kho `THANHHlv/PTIT-sentiment-analysis` được trang GitHub hiển thị **Public**
khi kiểm tra. Đừng đưa CSV thật hay ZIP nội bộ lên kho đó, kể cả Release,
Issue, Pull Request hoặc Git LFS. Repo private giới hạn người truy cập,
nhưng vẫn cần rà thông tin nhận diện và quyền chia sẻ trước khi đưa dữ liệu
lên. Với bản hiện tại, giữ code và dữ liệu tách riêng là lựa chọn phù hợp.

## Kết quả kiểm tra workspace

- Trước lần sửa này, Git theo dõi 45 file; một commit local `abff874`.
  Trong danh sách đó, `data/` chỉ có `.gitkeep`, template rỗng và dữ liệu
  tổng hợp ở `examples/`. Không có CSV thật hoặc checkpoint trong commit.
- Nhiều code mới/sửa chưa commit. Clone kho hiện tại chưa có đầy đủ code
  bàn giao, chẳng hạn `data/ai_dataset.py`, `data/annotation_review.py`,
  `preprocessing/phobert_input.py` và các kiểm tra bổ sung.
- Quy tắc cũ bỏ qua raw/labeled/splits, artifacts/outputs/cache, nhưng bỏ
  sót `data/chrome_screen.png`. Đã bổ sung bỏ qua file phát sinh tại gốc data,
  ZIP bàn giao, database, cookie/session theo tên thông dụng và khóa bí mật.
- Quét chuỗi khóa/token thông dụng trong file văn bản đang theo dõi hoặc
  có thể thêm vào Git không phát hiện chuỗi khớp. Đây là kiểm tra mẫu chuỗi,
  không phải chứng nhận mọi file hoàn toàn không có thông tin nhạy cảm.
- Quét text gốc: 12 dòng có chuỗi giống số điện thoại, 13 dòng có URL;
  trong 2.600 dòng có nhãn: 9 dòng giống số điện thoại và 11 dòng có URL.
  Không xuất các chuỗi đó vào tài liệu công khai. Cần người rà tiếp tên,
  nhắc tên, mã sinh viên, nội dung riêng tư; regex chưa chứng minh đã ẩn danh.
- `mask_sensitive: true` chỉ che khi tiền xử lý, không sửa CSV gốc, ZIP hay
  toàn bộ report. Text và URL nguồn vẫn có thể dẫn tới nhận diện người.
- `auto_annotate_workflow.py` tạo biến thể cross-check ngẫu nhiên, tự điền
  `confirmed=yes`; `create_resolutions.py` cũng tự điền xác nhận. Hai file
  giữ local để truy vết, bị bỏ qua khi thêm Git; không dùng như quy trình
  gán nhãn người. Các bản `assignments_v1/confirmed_v1`, `splits/real_v1`
  cũ không phải bản chuẩn đã được người xác nhận; ngày 07/10/2026 đã bỏ khỏi
  data và lưu riêng trong outputs/archives/data_cleanup_20261007.zip.
- `report/report.md` và `report/evidence/` chứa khẳng định xác nhận người,
  Kappa/đồng thuận và kết quả của bộ cũ. Đã tạm bỏ qua các file này; cần sửa
  nguồn nhãn, số liệu, kết luận và rà riêng tư trước khi gỡ quy tắc.
  Output cũ không được dùng làm kết quả của bộ chia mới.
- README hiện đã ghi đúng bộ nhãn AI; các tài liệu và lệnh lịch sử còn lại
  cần đọc cùng trạng thái nhãn và phiên bản, không suy thành kết quả hiện hành.

## Những gì nên commit

| File/thư mục | Xử lý |
|---|---|
| `src/ptit_sentiment/`, `tests/` | Code tái dùng và kiểm tra bằng fixture |
| `pyproject.toml`, `.gitignore`, `AGENTS.md`, `README.md` | Hướng dẫn, phụ thuộc và phiên bản hiện hành |
| `configs/classical.yaml`, `phobert*.yaml` | Cấu hình, đường dẫn tương đối, không credential |
| `docs/` | Hợp đồng dữ liệu, hướng dẫn nhãn, phân công, phương pháp, bàn giao; ghi rõ nội dung lịch sử |
| `data/examples/`, `data/labeled/template.csv`, `.gitkeep` | Dữ liệu tổng hợp và mẫu rỗng |
| `scripts/package_team_data.py`, `run_smoke.py`, `setup_vncorenlp.py`, CLI hỗ trợ đã rà | Chọn các công cụ cần tái lập |
| Cấu hình thu thập, script Chrome/CDP, probe/inspect/test trình duyệt | Chưa cần cho người 2–4; rà rồi chọn file cần dùng |
| Báo cáo và metrics nhỏ | Chỉ commit sau khi sửa nguồn nhãn, đối chiếu manifest/test, loại thông tin nhận diện |

Không commit `.venv/`, `.cache/` (gồm profile Chrome), `tools/vncorenlp/`,
cookie/token/password, ảnh DOM/màn hình, raw/snapshot/database, dữ liệu thật,
nhãn phân công và quyết định từng dòng, predictions/errors có text gốc,
logs/output lớn, joblib/checkpoint/PhoBERT hoặc ZIP bàn giao.
Máy khác cài thư viện từ `pyproject.toml`; tải VnCoreNLP/checkpoint theo
hướng dẫn, không sao chép cả môi trường máy người 1.

Gitignore không xóa file đã theo dõi hoặc đã xuất hiện trong lịch sử Git.
Không dùng `git add -f` để ép dữ liệu vào kho. Nếu sau này phát hiện dữ liệu
nhạy cảm đã push, cần xử lý lịch sử và quyền truy cập riêng; thêm ignore
không đủ. Git LFS phục vụ file lớn, không phải cách ẩn dữ liệu công khai.

## Phiên bản dữ liệu bàn giao

Nguồn: `data/labeled/ai_labeling_report.json` và `data/splits/manifest.json`,
đã kiểm tra bằng `load_split_set`, không thu thập lại hoặc train trong lần rà này.

| Tập | Số dòng | positive | neutral | negative |
|---|---:|---:|---:|---:|
| Train | 1.801 | 254 | 1.246 | 301 |
| Validation | 409 | 58 | 283 | 68 |
| Test | 390 | 50 | 271 | 69 |
| Tổng có nhãn | 2.600 | 362 | 1.800 | 438 |

2.864 dòng gốc = 2.600 có nhãn + 194 unresolved + 70 bị loại.
Phiên bản `ai_initial_4e93595ef7d1_seed42`, seed 42; tỷ lệ thực tế
69,27%/15,73%/15,00%, theo bài và thành phần liên kết trùng/gần trùng.
Không giao ID, post_id, text chuẩn hóa hoặc các cặp gần trùng đã ràng buộc;
cả ba tập đủ ba nhãn. SHA-256 manifest:

```text
18628d70bbf19194f7fc410f65bb7e6aa77d03263862dfbde418037767a4ba34
```

Toàn bộ 2.600 nhãn là AI, `label_source=ai`, `annotation_status=ai_labeled`.
Chưa có nhãn người xác nhận. Đây là bàn giao dữ liệu AI ban đầu theo ngoại lệ
đã được cho phép, chưa phải hoàn tất bộ nhãn chuẩn do bốn người gán.
Test đo mức phù hợp với nhãn AI. Cần đợt gán nhãn người độc lập để kết luận
chất lượng thực tế; đổi nhãn phải tạo phiên bản mới và đánh giá lại.

## Người 1 tạo và chia sẻ gói

```powershell
# Chạy từ thư mục gốc; dùng môi trường đã cài package
.\.venv\Scripts\python.exe scripts/package_team_data.py
Get-FileHash "handoff/ai_initial_4e93595ef7d1_seed42.zip" -Algorithm SHA256
```

Công cụ kiểm tra bộ chia, hash nhãn nguồn, báo cáo gán nhãn và sự khớp nhau
của labeled CSV với ba tập; từ chối ghi đè ZIP đã có. Gói chứa các file
trực tiếp trong `data/splits/` cần để loader hoạt động, cùng
`labeled_comments.csv`, `ai_labeling_report.json`, `unresolved.csv`,
`excluded_comments.csv`, `id_reconciliation.csv` và `TEAM_DATA.json` với hash.
Không gồm các phiên bản cũ, raw, ảnh, database, cache hoặc model.

Gói local đã tạo và kiểm tra: 14 file, 493.315 byte. SHA-256 ZIP:

```text
417ccf8758ffe040282381913b4e935b47f4aafa11775d28fb7885f1ca062143
```

Đã giải nén sang thư mục khác, kiểm tra hash của mọi file và nạp lại bộ chia
thành công; thử tạo trùng ZIP bị từ chối, ZIP cũ giữ nguyên. Validate CSV
2.600 dòng thành công; raw vẫn khớp hash gốc. Quy tắc Git bỏ qua data/ZIP/
báo cáo cũ nhưng vẫn cho phép code, hướng dẫn, template và ví dụ tổng hợp.
Kiểm tra này dùng thư mục nhận mới trong cùng workspace, chưa kiểm chứng
cài thư viện từ đầu trên máy một thành viên khác.

ZIP giữ nguyên text và mã nguồn; **chỉ chia sẻ trong nhóm được phép sử dụng**.
Người 1 tải thủ công lên thư mục lưu trữ riêng, chẳng hạn Drive giới hạn
đúng tài khoản bốn thành viên; không bật liên kết ai cũng xem được và không
đặt URL truy cập riêng trong README public. Truyền kèm SHA-256 ZIP và tên
phiên bản qua kênh nội bộ. Lần kiểm tra này chỉ tạo gói local, chưa tải lên.
Người 1 giữ raw/chứng cứ nguồn để đối soát; người 2–4 không cần toàn bộ raw.
Đợt kiểm tra nhãn người dùng file phân công mới trong
`annotation_batches_ai_v1`, chia sẻ riêng bởi người 1 theo
`docs/annotation_review.md`; không dùng file phân công cũ.
Gói ZIP này phục vụ tiền xử lý/huấn luyện và không kèm phân công gán nhãn.
Người 1 điều phối merge các file xác nhận trên máy giữ đủ dữ liệu kiểm toán.
`scripts/verify_ai_dataset.py` kiểm toán raw/snapshot/checkpoint nên chỉ
chạy trên máy người 1; máy người 2–4 dùng validate và loader như dưới đây.

## Các thành viên nhận dữ liệu và làm việc

Sau khi code bàn giao đã được commit/push và ZIP được chia sẻ riêng:

```powershell
git clone https://github.com/THANHHlv/PTIT-sentiment-analysis.git
Set-Location PTIT-sentiment-analysis
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
# Tải ZIP riêng vào handoff/, so hash với giá trị người 1 gửi
Get-FileHash "handoff/ai_initial_4e93595ef7d1_seed42.zip" -Algorithm SHA256
# Trên clone mới: không dùng -Force ghi đè một bộ chia đang làm việc
Expand-Archive -LiteralPath "handoff/ai_initial_4e93595ef7d1_seed42.zip" -DestinationPath .
.\.venv\Scripts\python.exe -m ptit_sentiment.data.validate --input data/labeled/labeled_comments.csv
.\.venv\Scripts\python.exe -c "from ptit_sentiment.data.split import load_split_set; f,m,h=load_split_set('data/splits'); print(m['dataset_version'], h, {k:len(v) for k,v in f.items()})"
```

Manifest chứa đường dẫn source tuyệt đối của máy tạo để truy vết; loader
đọc CSV/report tại thư mục được truyền vào, không yêu cầu máy mới có đường
dẫn đó. Không sửa manifest để đổi đường dẫn vì sẽ làm đổi hash so sánh.

| Thành viên | Tiếp tục từ bản bàn giao |
|---|---|
| Người 1 | Quản lý phiên bản, rà riêng tư, điều phối nhãn người/unresolved, đối soát |
| Người 2 | Tiền xử lý/BoW/TF-IDF; fit vocabulary và IDF trên train; không sửa text gốc |
| Người 3 | Bốn mô hình truyền thống và baseline; chọn cấu hình bằng validation |
| Người 4 | Chuẩn bị PhoBERT, fine-tune cùng bộ chia, CLI và tích hợp |

Các lệnh dưới đây là bước tiếp theo, **chưa chạy train trong lần kiểm tra này**:

```powershell
# Người 3: train và chọn bằng validation, chưa đánh giá test
.\.venv\Scripts\python.exe -m ptit_sentiment.training.train_classical --splits-dir data/splits --config configs/classical.yaml --artifact-dir artifacts/ai_initial/classical --output-dir outputs/ai_initial/training
# Người 4: cài riêng rồi kiểm tra tài nguyên; fine-tune theo README sau đó
.\.venv\Scripts\python.exe -m pip install -e ".[phobert]"
.\.venv\Scripts\python.exe -m ptit_sentiment.training.check_phobert --config configs/phobert_gpu4gb.yaml --output outputs/ai_initial/phobert_preflight.json
```

Không dùng `scripts/run_real.py` cho bộ AI hiện tại: script này yêu cầu
`human_labels_confirmed=true`. Dùng CLI từng mô hình có loader hỗ trợ
`ai_initial`; không chỉnh metadata thành xác nhận người để vượt kiểm tra.
Chốt cấu hình/checkpoint trước đánh giá test. Kết quả từ `real_v1` cũ không
được ghép vào so sánh bộ hiện tại. Mỗi người làm nhánh riêng và gửi PR,
không commit data/model/output cá nhân. Dữ liệu thay đổi do người 1 phát
hành gói version mới cho cả nhóm, thay vì mỗi người chia lại.

## Kiểm tra trước commit

```powershell
git status --short
git ls-files data artifacts outputs
git ls-files -ci --exclude-standard
git add .gitignore AGENTS.md pyproject.toml README.md src tests docs
git add configs/classical.yaml configs/phobert.yaml configs/phobert_gpu4gb.yaml configs/phobert_smoke.yaml
git add scripts/package_team_data.py scripts/run_smoke.py scripts/setup_vncorenlp.py
git diff --cached --stat
git diff --cached --name-only
git diff --cached
```

Đây là danh sách để người dùng tự rà và commit; chưa chạy git add/commit/push
trong lần này. Chọn thêm script cần dùng (như `verify_ai_dataset.py`) sau
khi đọc, tránh `git add .` khi còn hàng chục script thử trình duyệt.
Báo cáo chính cần sửa riêng trước công khai. Thêm ba thành viên làm
collaborator; kho private cũng cho nhóm làm việc khi cấp đúng quyền.

## Tài liệu GitHub đã đối chiếu

- [Kho dự án và trạng thái Public](https://github.com/THANHHlv/PTIT-sentiment-analysis).
- [Visibility và quyền truy cập](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories).
- [Gitignore không áp dụng cho file đã theo dõi](https://docs.github.com/en/get-started/git-basics/ignoring-files).
- [Git LFS và cách lưu file lớn](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).
