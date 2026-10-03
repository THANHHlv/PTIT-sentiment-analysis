# Khung báo cáo bài tập lớn

Đề tài: Phân loại cảm xúc bình luận sinh viên PTIT trên Facebook group.
Thành viên/lớp/giảng viên/ngày: [ĐIỀN THÔNG TIN THẬT].

## 1. Giới thiệu bài toán
Bối cảnh, mục tiêu ba nhãn, phạm vi group.
[ĐIỀN NGUỒN, THỜI GIAN, PHẠM VI THẬT]. Không suy rộng cho toàn bộ sinh viên.

## 2. Cơ sở lý thuyết
Sentiment classification, tiếng Việt, BoW/TF-IDF, n-gram, MultinomialNB,
LinearSVC, Transformer/RoBERTa/PhoBERT, baseline và metric đánh giá.

## 3. Dữ liệu và gán nhãn
Schema, lấy mẫu được phép, bảo vệ dữ liệu, nhãn và ví dụ ẩn danh.
[CHƯA CÓ SỐ LIỆU THẬT: số bình luận/bài viết, loại/trùng/cần xem xét].
[ĐIỀN số người gán, quy trình phân xử, bất đồng/kappa nếu đã đo].
Mô tả chia post_id, seed, hash, tỷ lệ thực và kiểm tra rò rỉ.

| Tập | Bình luận | Bài viết | positive | neutral | negative |
|---|---|---|---|---|---|
| Train | [CHƯA ĐO] | [CHƯA ĐO] | [CHƯA ĐO] | [CHƯA ĐO] | [CHƯA ĐO] |
| Validation | [CHƯA ĐO] | [CHƯA ĐO] | [CHƯA ĐO] | [CHƯA ĐO] | [CHƯA ĐO] |
| Test | [CHƯA ĐO] | [CHƯA ĐO] | [CHƯA ĐO] | [CHƯA ĐO] | [CHƯA ĐO] |

## 4. Tiền xử lý và đặc trưng
Ví dụ trước/sau, NFC/URL/slang, phủ định, emoji/stop words, tách từ.
Vocabulary/IDF chỉ train; n-gram; đường xử lý PhoBERT riêng.
[ĐIỀN ví dụ thật ẩn danh và số đặc trưng thực tế].

## 5. Mô hình và cấu hình thực nghiệm
Baseline, 4 tổ hợp, PhoBERT; lưới alpha/C/n-gram và chọn validation Macro-F1;
checkpoint/epoch, seed, phần cứng, thời gian, phiên bản, manifest.
[ĐIỀN CẤU HÌNH ĐÃ CHẠY; không mô tả dự kiến như đã thực nghiệm].

## 6. Kết quả và so sánh
Chỉ điền metrics.json test thật cùng manifest; không dùng synthetic.csv cho bảng này.

| Mô hình | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|
| Majority baseline | [CHƯA CHẠY THẬT] | — | — | — |
| BoW + NB | [CHƯA CHẠY THẬT] | — | — | — |
| TF-IDF + NB | [CHƯA CHẠY THẬT] | — | — | — |
| BoW + SVM | [CHƯA CHẠY THẬT] | — | — | — |
| TF-IDF + SVM | [CHƯA CHẠY THẬT] | — | — | — |
| PhoBERT | [CHƯA FINE-TUNE] | — | — | — |

[CHÈN per-label P/R/F1/support, confusion matrix của mô hình đã chạy].
[VIẾT NHẬN XÉT SAU KẾT QUẢ THẬT; chưa kết luận mô hình tốt nhất].
Nêu giới hạn một seed/test nhỏ nếu chênh lệch thấp.

## 7. Phân tích lỗi
[VÍ DỤ THẬT TỪ errors.csv ĐÃ ẨN DANH], nhãn thật/dự đoán, ngữ cảnh và giả thuyết nguyên nhân.
Phủ định, mỉa mai, slang, hỗn hợp, thiếu ngữ cảnh, tách từ, truncation, nhãn nghi vấn.
[ĐIỀN thống kê lỗi đã xem; phân biệt giả thuyết và bằng chứng].

## 8. Hạn chế và kết luận
[ĐIỀN KẾT LUẬN DỰA TRÊN THỰC NGHIỆM THẬT].
Đại diện dữ liệu, nhãn, phần cứng, mất cân bằng, rò rỉ còn lại, thiếu ngữ cảnh, áp dụng ngoài group.
Hướng phát triển (đề xuất): nhiều nguồn/seed, hiệu chỉnh xác suất, gán nhãn bổ sung.

## 9. Tài liệu tham khảo
Nguyen & Nguyen (2020), PhoBERT; sklearn, Underthesea, Transformers thực dùng,
VnCoreNLP, nguồn và quy định sử dụng dữ liệu.
[CHUẨN HÓA TRÍCH DẪN VÀ NGÀY TRUY CẬP TRƯỚC NỘP].
