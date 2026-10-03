# Kiểm chứng bàn giao

Ngày môi trường: 2026-10-02. Workspace ban đầu trống; không tìm thấy AGENTS.md
trong dự án hoặc các thư mục cha D:/, D:/PYTHON/, D:/PYTHON/NLP/.
Không có code cũ bị ghi đè.

## Đã chạy thành công

Lần kiểm tra: smoke_verified_v2, provenance=synthetic.
60 bình luận tự tạo, 20 bài viết giả định; không lấy dữ liệu Facebook.

| Tập | Bình luận | post_id | positive | neutral | negative |
|---|---|---|---|---|---|
| Train | 42 | 14 | 14 | 14 | 14 |
| Validation | 9 | 3 | 3 | 3 | 3 |
| Test | 9 | 3 | 3 | 3 | 3 |

- Validate, chia nhóm seed=42, đủ ba nhãn, tỷ lệ 70/15/15 đúng trên mẫu này.
- Train BoW+NB, TF-IDF+NB, BoW+SVM, TF-IDF+SVM; mỗi mô hình thử 4 cấu hình.
- Baseline nhãn phổ biến nhất từ train.
- Lưu/tải bốn pipeline joblib và đối chiếu dự đoán.
- Evaluate test cả 5 artifact, xuất JSON/CSV metric, per-label/support,
  confusion matrix PNG/CSV, predictions và errors.
- Compare từ các metrics.json thực sự được xuất.
- Predict một câu ra CSV, một CSV 3 câu mới, và một câu trả JSON trên terminal.
- Kiểm tra không giao post_id/id/khóa bình luận, text gốc được bảo toàn,
  SVM không có cột probability; NB ánh xạ probability theo đúng tên nhãn.
- 13 kiểm tra unittest đạt (chống trùng/rò rỉ, dữ liệu sai/nhỏ, hash,
  phủ định, vocabulary không đổi khi transform, metric đủ nhãn,
  artifact thiếu, compare/evaluate khác bộ chia).
- Kiểm tra PhoBERT bằng mock segmenter: giữ hoa thường/slang/URL và phục hồi cwd;
  kiểm tra max_length sai trước import/tải nặng. Đây không phải kiểm chứng JVM/model thật.
- compileall không báo lỗi; CLI train_phobert --help chạy.
- Import module PhoBERT không nạp torch, transformers hoặc py_vncorenlp.
- Đã xem PNG confusion matrix baseline: tên nhãn rõ, trục hàng thật/cột dự đoán đầy đủ.

Nhật ký lệnh/exit code: outputs/smoke_verified_v2/commands.json.
Chứng nhận kiểm tra: outputs/smoke_verified_v2/verification.json.
Bảng metric tổng hợp: outputs/smoke_verified_v2/comparison.csv.
Bộ chia: data/splits/smoke_verified_v2/manifest.json.
Artifact: artifacts/smoke_verified_v2/.
Không đưa các điểm số tổng hợp này vào khung báo cáo thật.

Ví dụ preprocessing thực chạy:
- Gốc: “  SV ko  thích lịch học 😢 https://example.org  ”
- Normalize: “sinh viên không thích lịch học 😢 urltoken”
- Tokenize: “sinh_viên không thích lịch_học 😢 urltoken”

## Môi trường thực kiểm tra

Python 3.12.9; scikit-learn 1.5.2; pandas 2.2.3; numpy 1.26.4;
PyYAML 6.0.2; joblib 1.4.2; matplotlib 3.9.2; underthesea 9.5.0.
Dùng .venv --system-site-packages tái dùng thư viện sẵn có;
package được cài editable bằng --no-deps --no-build-isolation.
Chưa kiểm chứng cài mới hoàn toàn từ Internet. README cung cấp lệnh cài thông thường.

## Chưa chạy và việc còn cần làm

Không tải checkpoint PhoBERT hoặc VnCoreNLP; không chạy setup tải tài nguyên;
không khởi tạo Java/JVM thật; không fine-tune, evaluate hoặc predict PhoBERT thật.
API đối chiếu [PhoBERT](https://github.com/VinAIResearch/PhoBERT),
[Trainer 4.44.2](https://huggingface.co/docs/transformers/v4.44.2/main_classes/trainer),
[VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) và
[wrapper chính thức được VnCoreNLP dẫn tới](https://github.com/thelinhbkhn2014/VnCoreNLP_Wrapper).
Người 4 phải kiểm chứng optional dependencies, Java/pyjnius và phần cứng,
ghim checkpoint/segmenter commit, rồi chạy cùng manifest dữ liệu thật.

Nhóm cần dữ liệu thật được phép sử dụng, gán nhãn, rà soát needs_review/gần trùng,
chốt bộ chia và chạy thực nghiệm trước khi điền báo cáo.
Dữ liệu mẫu có template lặp theo chủ đề, nhãn dễ tách và rất nhỏ;
không chứng minh chất lượng tổng quát hoặc thứ hạng các mô hình.
