# Phân công và bàn giao

Cả nhóm dùng cùng package, [hợp đồng dữ liệu](data_contract.md), labels.py và manifest.
Mỗi người làm nhánh riêng trong cùng kho; không tự chia lại dữ liệu trên từng máy.

| Người | Phần phụ trách | Đầu vào | Đầu ra cần bàn giao |
|---|---|---|---|
| 1 | data/, điều phối gán nhãn, ghép báo cáo | Bình luận thật và hồ sơ nguồn, nhãn thành viên | CSV nhãn, bộ chia/manifest, thống kê nguồn/nhãn, phần báo cáo dữ liệu |
| 2 | preprocessing/, features/ | Train, text gốc, classical.yaml | Transformer lưu/tải được, BoW/TF-IDF, ví dụ trước/sau, giải thích lựa chọn |
| 3 | models/classical.py, training/train_classical.py, evaluation/ | Bộ chia khóa và tiền xử lý | Bốn pipeline + baseline, log chọn tham số, metrics/ma trận/lỗi, bảng so sánh |
| 4 | PhoBERT, training/train_phobert.py, predict.py | Cùng bộ chia, checkpoint, Java/VnCoreNLP | Model/tokenizer/config, log/truncation, kết quả cùng schema, CLI và hướng dẫn chạy |

## Điều kiện bàn giao

- Người 1: validate đạt, mỗi tập đủ ba nhãn; không giao ID/bài/nội dung;
  truy vết nguồn/seed/hash và ghi đúng nguồn nhãn. Đợt rà người theo [hướng dẫn nhãn](labeling_guide.md).
- Người 2: giữ phủ định/dấu hiệu cảm xúc, không ghi đè text gốc;
  vocabulary/IDF chỉ fit train. Phối hợp người 3 về pipeline, người 4 về đầu vào PhoBERT riêng.
- Người 3: chọn alpha/C/n-gram bằng Macro-F1 validation, chốt trước test;
  lưu/tải giữ dự đoán, đủ độ đo ba nhãn; không báo decision score SVM thành xác suất.
- Người 4: fine-tune thật, chọn checkpoint bằng validation, lưu thống kê cắt token;
  predict không train lại. Smoke một bước chưa thay thế huấn luyện đầy đủ.

Cả bốn rà nhãn độc lập, thảo luận bất đồng sau khi lưu nhãn riêng và cùng đọc lỗi.
Người 1 ghép báo cáo; người 3 và 4 tổng hợp kết quả trên cùng manifest/test.
Báo cáo phải ghi rõ nhãn AI, dữ liệu tổng hợp và thực nghiệm thật; không tự điền Kappa hoặc điểm chưa đo.

## Nhận và chia sẻ dữ liệu

Người 1 đóng gói bộ tối giản bằng scripts/package_team_data.py theo [hợp đồng dữ liệu](data_contract.md).
Thành viên clone code, cài package theo [README](../README.md), nhận gói dữ liệu riêng,
sao chép data/ vào gốc clone rồi chạy verify/validate. Không thu thập lại để có cùng bộ đầu vào.
Chỉ commit code/cấu hình/tests/tài liệu; không đưa dữ liệu Facebook thật lên kho công khai.
Snapshot cũ trong handoff/ không tự cập nhật khi sửa workspace; phát hành gói mới nếu cần bàn giao lại.

## Báo cáo và phần tiếp theo

Viết tại [report/report.md](../report/report.md): giới thiệu → lý thuyết → dữ liệu/gán nhãn
→ tiền xử lý → thiết kế thực nghiệm → kết quả → phân tích lỗi → hạn chế/kết luận → tài liệu tham khảo.
Bản báo cáo cũ cần rà các khẳng định xác nhận người/Kappa và phiên bản kết quả trước khi dùng.
Chỉ đưa số liệu truy vết được về output thực tế; phần chưa chạy phải ghi rõ.

Bộ AI hiện tại đã chốt 2.600 dòng, chưa có nhãn người xác nhận.
Bước thực nghiệm tiếp theo: người 2 chốt tiền xử lý; người 3/4 huấn luyện và
đánh giá trên cùng bộ chia. Đây là việc tiếp theo, không chạy trong nhiệm vụ dọn tài liệu.
Xem [phương pháp](methodology.md) và [hướng dẫn học](learning_guide.md) để giải thích phần mình.
