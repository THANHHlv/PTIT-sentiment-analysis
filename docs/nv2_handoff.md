# Bàn giao nhiệm vụ 2 — Tiền xử lý và trích chọn đặc trưng

Cập nhật: 07/10/2026  
Phạm vi: phần tiền xử lý văn bản và tạo đặc trưng BoW/TF-IDF. Tài liệu này giúp thành viên tiếp theo biết NV2 đã làm gì, dùng file nào, đã kiểm thử đến đâu và còn vướng gì.

## 1. Đầu vào đã sử dụng

NV2 nhận ba tập do NV1 bàn giao:

| Tập | Số dòng | Cách NV2 sử dụng |
|---|---:|---|
| `data/splits/train.csv` | 1.801 | Fit bộ tiền xử lý, vocabulary và IDF. |
| `data/splits/validation.csv` | 409 | Chỉ transform bằng bộ đã fit từ train. |
| `data/splits/test.csv` | 390 | Chỉ transform bằng bộ đã fit từ train. |

NV2 chỉ sử dụng cột `text` để tạo đặc trưng. Cột `label` không tham gia việc xây dựng vocabulary hoặc IDF. Các CSV gốc không bị ghi đè.

Bộ 2.600 dòng hiện mang nhãn AI, gồm 1.800 `neutral`, 438 `negative`, 362 `positive`. Đây chưa phải bộ nhãn người xác nhận độc lập.

## 2. Những việc đã thực hiện

### 2.1. Chuẩn hóa văn bản

Đã hoàn thiện và kiểm tra các bước:

- Chuẩn hóa Unicode NFC và khoảng trắng.
- Loại BOM, zero-width space, ký tự điều khiển và ký tự lỗi `U+FFFD`.
- Chuyển chữ thường theo cấu hình.
- Thay URL bằng `urltoken` và giữ dấu câu ngay sau URL.
- Che email, số điện thoại và `@handle` bằng các token chung.
- Mở rộng một số tiếng lóng rõ nghĩa theo bảng cấu hình.
- Báo lỗi rõ khi đầu vào không phải chuỗi hoặc bình luận mới rỗng.

Code chính:

- `src/ptit_sentiment/preprocessing/normalize.py`
- `src/ptit_sentiment/preprocess.py`

`preprocess.py` là API công khai. Hàm `preprocess_text()` xử lý trực tiếp một bình luận và dùng cùng implementation với pipeline huấn luyện/dự đoán.

### 2.2. Tách từ tiếng Việt

Đã dùng Underthesea 9.5.0 để tách từ. Từ nhiều âm tiết được nối bằng dấu gạch dưới, ví dụ:

```text
Gốc:       SV ko thích lịch học 😢 https://example.org
Chuẩn hóa: sinh viên không thích lịch học 😢 urltoken
Tách từ:   sinh_viên không thích lịch_học 😢 urltoken
```

Code chính: `src/ptit_sentiment/preprocessing/tokenize.py`.

### 2.3. Emoji, stop words và phủ định

Lựa chọn hiện tại trong `configs/classical.yaml`:

- `emoji: keep`: giữ emoji vì emoji có thể mang sắc thái cảm xúc.
- `stop_words: []`: chưa loại stop words để tránh mất ngữ cảnh trên tập dữ liệu nhỏ.
- Bắt buộc giữ các từ phủ định: `không`, `chưa`, `chẳng`, `chả`, `đừng`.
- Nếu stop words chứa từ/cụm có phủ định, `TextPreprocessor` báo lỗi.

Bảng slang và bảng đổi emoji là quy tắc thủ công, không phải từ điển tự học từ dữ liệu. Cấu hình hiện giữ emoji nên bảng đổi emoji thành từ không được áp dụng trong lần xử lý hiện tại. Các ánh xạ ngắn như `k`, `dc` có thể đa nghĩa và cần rà ngữ cảnh trước khi mở rộng thêm.

### 2.4. BoW và TF-IDF

Đã xây dựng:

- BoW bằng `CountVectorizer`.
- TF-IDF bằng `TfidfVectorizer`.
- Unigram: `ngram_range=(1, 1)`.
- Unigram + bigram: `ngram_range=(1, 2)`.
- Hỗ trợ `min_df`, `max_df`, `max_features` từ YAML.
- Giữ token đã được Underthesea tách, emoji và dấu câu qua tokenizer khoảng trắng.

Code chính:

- `src/ptit_sentiment/features/vectorizers.py`: tạo từng vectorizer.
- `src/ptit_sentiment/features/__init__.py`: API `make_vectorizer`.
- `features.py` ở gốc dự án: file bàn giao tạo đồng thời ma trận train, validation, test và vector câu mới.

### 2.5. Chống rò rỉ dữ liệu

Hàm `build_feature_matrices()` trong `features.py` thực hiện đúng thứ tự:

1. `fit_transform()` tiền xử lý và vectorizer trên train.
2. `transform()` validation bằng các đối tượng đã fit.
3. `transform()` test bằng các đối tượng đã fit.
4. `transform_comment()` xử lý bình luận mới mà không fit lại.

Đã kiểm tra vocabulary và IDF không đổi sau khi transform validation, test hoặc câu mới. Token chỉ xuất hiện ngoài train không tạo cột mới.

## 3. Kết quả chạy trên dữ liệu hiện tại

| Đặc trưng | Train | Validation | Test | Vector 0 ở validation/test |
|---|---:|---:|---:|---:|
| BoW unigram | `(1801, 3543)` | `(409, 3543)` | `(390, 3543)` | 9 / 4 |
| BoW unigram + bigram | `(1801, 17672)` | `(409, 17672)` | `(390, 17672)` | 9 / 4 |
| TF-IDF unigram | `(1801, 3543)` | `(409, 3543)` | `(390, 3543)` | 9 / 4 |
| TF-IDF unigram + bigram | `(1801, 17672)` | `(409, 17672)` | `(390, 17672)` | 9 / 4 |

Các con số trên mô tả đầu ra đặc trưng, không phải điểm phân loại. Chín câu validation và bốn câu test có vector toàn số 0 vì không còn token nào thuộc vocabulary train sau xử lý.

## 4. File đã thêm hoặc cập nhật cho NV2

| File | Nội dung |
|---|---|
| `src/ptit_sentiment/preprocess.py` | API xử lý một câu và xuất các thành phần tiền xử lý dùng chung. |
| `src/ptit_sentiment/preprocessing/normalize.py` | Chuẩn hóa, URL, ký tự lỗi, thông tin nhạy cảm, slang và emoji. |
| `src/ptit_sentiment/preprocessing/tokenize.py` | Underthesea, stop words và bảo vệ phủ định. |
| `src/ptit_sentiment/features/vectorizers.py` | Tạo BoW/TF-IDF, kiểm tra unigram/bigram. |
| `src/ptit_sentiment/features/__init__.py` | API công khai cho đặc trưng. |
| `features.py` | File bàn giao: fit train, transform validation/test/câu mới. |
| `tests/test_nv2.py` | Test riêng cho toàn bộ hợp đồng NV2. |
| `docs/nv2_examples.md` | Sáu ví dụ trước/sau và minh họa vector BoW/TF-IDF. |
| `docs/nv2_plan.md` | Kế hoạch, quy tắc và nhật ký triển khai NV2. |
| `report/nv2_preprocessing_features.md` | Phần báo cáo lý thuyết và kết quả kiểm tra NV2. |

## 5. Kiểm thử đã chạy

Lần kiểm tra gần nhất:

- 12/12 test riêng NV2 đạt.
- 54/54 test toàn dự án đạt.
- Underthesea thật đã chạy trên sáu câu minh họa.
- BoW/TF-IDF đã chạy trên cả ba CSV hiện tại.
- CSV không thay đổi sau khi tạo đặc trưng.

Chạy lại trên PowerShell:

```powershell
cd D:\Project\PTIT-sentiment-analysis
$env:PYTHONIOENCODING = "utf-8"

# Test riêng NV2
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_nv2.py -v

# Test toàn dự án
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 6. Cách người tiếp theo sử dụng NV2

```python
import pandas as pd
import yaml
from features import build_feature_matrices

with open("configs/classical.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

train = pd.read_csv("data/splits/train.csv", dtype={"id": str, "post_id": str})
validation = pd.read_csv("data/splits/validation.csv", dtype={"id": str, "post_id": str})
test = pd.read_csv("data/splits/test.csv", dtype={"id": str, "post_id": str})

bundle = build_feature_matrices(
    train["text"],
    validation["text"],
    test["text"],
    kind="tfidf",
    ngram_range=(1, 2),
    preprocessing_options=config["preprocessing"],
    vectorizer_options=config["vectorizer"],
)

print(bundle.X_train.shape)
print(bundle.X_validation.shape)
print(bundle.X_test.shape)
new_vector = bundle.transform_comment("Mình chưa hài lòng với lịch học")
```

Trong luồng huấn luyện chính, thành viên phụ trách mô hình nên dùng pipeline trong `src/ptit_sentiment/models/classical.py`. Pipeline lưu chung tiền xử lý, vectorizer và classifier để dự đoán câu mới không bị lệch so với lúc train.

## 7. Kiến thức cần nắm khi bảo vệ

- Văn bản gốc được chuẩn hóa, tách từ, sau đó mỗi token/n-gram tương ứng với một cột vector.
- BoW lưu số lần token xuất hiện.
- TF-IDF tăng vai trò của token có tính phân biệt và giảm vai trò của token xuất hiện trong nhiều tài liệu; sklearn còn chuẩn hóa vector L2 theo mặc định.
- Không fit vocabulary hoặc IDF trên validation/test vì như vậy đã dùng thông tin của tập đánh giá trước khi đánh giá, gây rò rỉ dữ liệu.
- Validation dùng chọn cấu hình; test chỉ dùng đánh giá sau khi cấu hình đã chốt.

## 8. Việc còn vướng và chưa thuộc kết quả hoàn thành

`data/splits/train.csv`, `validation.csv`, `test.csv` và `data/labeled/labeled_comments.csv` hiện không khớp SHA-256 ghi trong `manifest.json`. `scripts/verify_ai_dataset.py` vì vậy dừng ở bước kiểm tra hash. NV2 không sửa CSV hoặc manifest để né kiểm tra này. NV1 cần cung cấp đúng cặp CSV/manifest hoặc xác nhận cách tạo lại manifest từ nguồn đã kiểm toán.

NV2 chưa huấn luyện NB/SVM/PhoBERT và chưa tạo điểm Accuracy/F1; đó là phần huấn luyện/đánh giá. `artifacts/` và `outputs/` hiện chưa chứa kết quả của bộ dữ liệu hiện tại. Phần báo cáo NV2 đang ở file riêng, chưa ghép vào báo cáo tổng.

## 9. Kết luận bàn giao

Phần tiền xử lý và trích chọn đặc trưng đã được code, chạy trên dữ liệu hiện tại và kiểm thử. Người tiếp theo có thể dùng trực tiếp API và cấu hình đã nêu. Trước khi chạy thực nghiệm chính thức và công bố điểm mô hình, cần giải quyết lỗi không khớp hash của bộ dữ liệu với manifest.
