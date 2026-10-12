# Kiểm tra bàn giao người 2 — 09/10/2026

## Phạm vi và kết luận

Phần NV2 **đạt điều kiện tích hợp huấn luyện** trên bộ
`ai_initial_4e93595ef7d1_seed42`. Giữ nguyên tất cả CSV, nhãn, manifest và bộ chia.
Nhánh kiểm tra: `thanhlv/demo`, HEAD tại đầu lần chạy `9a72911`.
Workspace đã có thay đổi chưa commit ở README, nhiều tài liệu, collect.py và
prepare_team_handoff.py; không checkout/merge/reset/commit/push, giữ các thay đổi đó.

## File đã đọc

- `AGENTS.md`, `README.md`, `pyproject.toml`, `.gitignore`.
- `docs/nv2_handoff.md`, `docs/task_assignment.md`, `docs/methodology.md`, tài liệu hợp đồng/nhãn.
- `src/ptit_sentiment/preprocess.py`, `preprocessing/normalize.py`, `preprocessing/tokenize.py`.
- `src/ptit_sentiment/features/vectorizers.py`, `features/__init__.py`, `features.py` ở gốc.
- `tests/test_nv2.py`, kiểm thử hợp đồng và các module dữ liệu, pipeline, train/evaluate/predict.
- `data/labeled/labeled_comments.csv`, ba CSV chia và `data/splits/manifest.json`.

Ba file được liệt kê trong bàn giao nhưng **không có tại workspace**:
`docs/nv2_examples.md`, `docs/nv2_plan.md`, `report/nv2_preprocessing_features.md`.
Không tái tạo chúng rồi coi là bản NV2 đã kiểm tra. Implementation và test NV2 có đủ.
Không thấy bộ ma trận/vectorizer lưu riêng cho bộ chia hiện tại; tạo mới đặc trưng từ code NV2,
không lấy artifact cũ `real_v1` làm đầu vào.

## Kiểm tra thực sự chạy

Lệnh chạy có thể tái lập:

```powershell
.\.venv\Scripts\python.exe scripts/review_part2.py --output outputs/part2_review_new.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Bằng chứng đầy đủ của lần chính:
`outputs/part3_20261009_cpu/part2_audit.json`, `commands.json`, `verification.json`.
54 test ban đầu đạt; sau bổ sung kiểm thử NV3 đạt **57/57**, gồm **12 test NV2**.
Audit dùng Underthesea thật 9.5.0, không mock tokenizer.

| Nội dung | Kết quả |
|---|---|
| Hash CSV / manifest | Cả bốn CSV khớp manifest và khớp byte trong ZIP `data_minimal_20261007.zip` |
| Labeled so với tổng ba tập | Khớp mọi cột, ID, text, nhãn và metadata |
| Schema / ô rỗng / nhãn / trùng | Validator đạt; ba nhãn đúng trong từng tập |
| Giao ID / post_id / text chuẩn hóa | Không giao; kiểm tra cả ràng buộc nhóm và gần trùng hiện có |
| Nguồn nhãn | 2.600 `label_source=ai`, `annotation_status=ai_labeled`; 0 nhãn người xác nhận |
| Unicode / khoảng trắng / URL / slang | Test chuẩn hóa đạt; chạy ví dụ thật qua Underthesea |
| Phủ định / emoji / dấu câu | Giữ phủ định, emoji, dấu câu; từ phủ định không được nằm trong stop words |
| Stop words | Cấu hình `[]`; kiểm tra kiểu và bảo vệ phủ định |
| Vocabulary | Mọi n-gram đều có nguồn từ train; token ngoài train không thêm cột |
| IDF | Đối chiếu trực tiếp công thức IDF với DF đếm từ train; không đổi sau transform heldout |
| Sparse / không âm | Cả 12 ma trận thuộc bốn cấu hình đều đạt |
| Thứ tự vector | So từng dòng đầu/giữa/cuối với transform riêng bình luận cùng ID; giữ nguyên số dòng |
| CSV gốc | SHA-256 trước/sau audit và toàn bộ thực nghiệm không đổi |

| Đặc trưng | Train | Validation | Test | Vector 0: train / val / test |
|---|---|---|---|---|
| BoW unigram | 1801×3543 | 409×3543 | 390×3543 | 0 / 9 / 4 |
| BoW unigram–bigram | 1801×17672 | 409×17672 | 390×17672 | 0 / 9 / 4 |
| TF-IDF unigram | 1801×3543 | 409×3543 | 390×3543 | 0 / 9 / 4 |
| TF-IDF unigram–bigram | 1801×17672 | 409×17672 | 390×17672 | 0 / 9 / 4 |

`TextPreprocessor.fit` không học tham số; transform mỗi bình luận đúng một lần trong đường inference.
Pipeline nhận text gốc, không nhận output đã tách từ. NV3 tách từ train/validation một lần dùng chung
giữa các ứng viên để tiết kiệm CPU; vectorizer/classifier từng ứng viên vẫn fit riêng trên train.

## Vấn đề, ảnh hưởng và sửa tối thiểu

1. Cảnh báo hash ở mục 8 của `nv2_handoff.md` **không tái hiện ở workspace hiện tại**.
   Không sửa manifest hoặc CSV để bỏ qua kiểm tra. Manifest hiện có hash
   `c743878f73ef20ef786f46f46c62511b894a10891c2893de8d75cb753522e4e3`;
   hash byte manifest có thể khác bản ZIP do định dạng xuống dòng, metadata nhúng vẫn qua kiểm tra cấu trúc.
2. Pipeline cũ có thể đưa câu rỗng sau làm sạch vào vectorizer rồi âm thầm dự đoán.
   Sửa nhỏ trong `TextPreprocessor.transform`: báo `ValueError` có vị trí khi đầu vào thiếu/rỗng,
   rỗng sau chuẩn hóa hoặc sau tách từ/stop words. Đã kiểm tra None, chuỗi rỗng, khoảng trắng,
   chỉ ký tự lỗi/vô hình, và emoji bị loại hết. Không có câu dữ liệu chính bị loại bởi sửa này.
3. Câu hợp lệ nhưng ngoài vocabulary vẫn được giữ thành vector 0; đây là hạn chế đặc trưng,
   không đổi nhãn hoặc loại mẫu để tăng điểm. NB/SVM vẫn dự đoán được.
4. Slang `k`, `dc` có thể đa nghĩa; Underthesea đôi lúc nối emoji với token cạnh nó,
   ví dụ audit có `😢_urltoken`. Ghi nhận hạn chế, không sửa từ điển theo test.
5. Che email/điện thoại/@handle không đảm bảo loại hết tên người trong text.
   Ví dụ đưa vào phân tích lỗi đã được rà riêng; CSV dự đoán đầy đủ chỉ giữ local trong outputs.

Không viết lại phần NV2. Các bổ sung thời gian, hội tụ, khóa lựa chọn, hash artifact thuộc tích hợp NV3;
xem [bàn giao phần 3](part3_handoff.md).
