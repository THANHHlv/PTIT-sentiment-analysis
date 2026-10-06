# Phân công bốn người

Dùng chung package src, labels.py, data_contract.md và manifest. Làm nhánh riêng theo phần.
Đổi giao tiếp phải thống nhất, cập nhật tài liệu và smoke test. Không có slide/giao diện.

## Người 1 — Dữ liệu

- Việc: chuẩn bị dữ liệu thật được phép dùng, ẩn thông tin cá nhân, gán nhãn,
  giải quyết needs_review, rà trùng; hoàn thiện validate/split; khóa bộ chia nhóm.
- Đầu vào: bình luận nguyên văn, mã bài viết, labeling_guide và mẫu CSV.
- Đầu ra: labeled CSV, train/validation/test, manifest, thống kê nguồn/nhãn và nhật ký gán nhãn.
- Hoàn thành: validate pass; đủ ba nhãn mỗi tập, không giao id/post_id/text;
  tái lập seed/hash; có mô tả lấy mẫu và kiểm tra chất lượng nhãn.
- Báo cáo: dữ liệu, gán nhãn, chia tập, mất cân bằng và tính đại diện; điều phối ghép report/report.md, thống nhất thuật ngữ, số liệu và trích dẫn.
- Câu hỏi: vì sao chia theo bài viết? neutral khác không rõ thế nào?
  xử lý mỉa mai/bất đồng ra sao? còn rò rỉ gần trùng/người viết không?
  tỷ lệ thực tế lệch 70/15/15 bao nhiêu?

## Người 2 — Tiền xử lý và đặc trưng

- Việc: hoàn thiện normalize/tokenize/vectorizers; kiểm chứng Underthesea;
  rà slang/emoji/stop words; thử n-gram và lưu ví dụ trước/sau.
- Đầu vào: train, config, data contract, text gốc.
- Đầu ra: hàm/transformer sklearn tái dùng, cấu hình BoW/TF-IDF, tài liệu lựa chọn.
- Hoàn thành: giữ phủ định, không sửa CSV gốc; vocabulary/IDF chỉ fit train;
  transformer lưu/tải joblib được; API thống nhất với người 3; PhoBERT có đường riêng.
- Báo cáo: xử lý tiếng Việt, đặc trưng, n-gram và tham số.
- Câu hỏi: BoW khác TF-IDF thế nào? bigram hỗ trợ phủ định ra sao?
  vì sao không xóa mọi emoji/stop words? tách từ khác tokenizer subword thế nào?
  tại sao không fit trên validation/test?

## Người 3 — Truyền thống và đánh giá

- Việc: train 4 tổ hợp và baseline; lưới nhỏ chọn Macro-F1 validation;
  lưu pipeline; hoàn thiện metrics/evaluate/compare; đọc errors và chốt bảng thật.
- Đầu vào: bộ chia khóa, preprocessing/vectorizer, classical.yaml.
- Đầu ra: 5 artifact/metadata, validation search log, JSON/CSV metrics,
  confusion matrix PNG/CSV, predictions/errors và bảng so sánh.
- Hoàn thành: test không chọn tham số, cùng manifest; lưu/tải cùng dự đoán;
  đủ metric 3 nhãn; SVM không có probability giả; chỉ tổng hợp lần đã evaluate.
- Báo cáo: NB/SVM, tuning, baseline, kết quả truyền thống, lỗi.
- Câu hỏi: giả định NB? C/alpha tác động gì? vì sao Macro-F1?
  accuracy sai lệch khi nào? decision score khác probability thế nào?
  nhãn nào thường bị nhầm và có bằng chứng gì?

## Người 4 — PhoBERT và tích hợp

- Việc: optional dependencies/Java/VnCoreNLP; kiểm chứng segmentation;
  fine-tune cùng split, chọn checkpoint validation Macro-F1; lưu đầy đủ;
  hoàn thiện predict/README, chạy tích hợp và ghép kết quả thật.
- Đầu vào: bộ chia khóa, phobert.yaml, segmenter local và artifact classical.
- Đầu ra: PhoBERT artifact/trainer log, metric cùng schema, CLI và tài liệu được kiểm chứng.
- Hoàn thành: checkpoint/phiên bản tái lập; test tách khỏi train; predict không fit lại;
  classical không cần torch/transformers; các thành viên chạy smoke.
  PhoBERT đã kiểm chứng fine-tune một bước và lưu/tải/dự đoán; chưa chạy đầy đủ trên dữ liệu thật.
- Báo cáo: PhoBERT, cấu hình hiện đại, tích hợp, hạn chế/kết luận, rà tài liệu tham khảo.
- Câu hỏi: input PhoBERT cần gì? RDRSegmenter khác BPE thế nào?
  checkpoint chọn ở đâu? truncation ảnh hưởng câu dài ra sao?
  tài nguyên/thời gian thực bao nhiêu? so sánh có cùng test không?

## Tích hợp

Người 1 bàn giao bộ chia bất biến; người 2 bàn giao transformer đã kiểm tra;
người 3/4 chạy độc lập cùng bộ chia. Người 4 chạy smoke và compare các metrics thực tế.
Cả nhóm phân tích lỗi và hoàn thiện báo cáo, không điền số liệu giả.

## File bổ sung cho giai đoạn dữ liệu thật

- Người 1: configs/collection.yaml, data/collect.py, data/labeling.py, data/split.py; giữ chứng cứ nguồn và xác nhận nhãn.
- Cả bốn: annotator_1.csv đến annotator_4.csv; gán độc lập phần cross_check, chưa dùng suggested_label làm gold.
- Người 3: scripts/run_real.py train/evaluate, các module classical/training/evaluation; đọc docs/learning_guide.md về NB/SVM và độ đo.
- Người 4: scripts/probe_public_sources.py (chỉ kiểm tra HTTP), configs/phobert_gpu4gb.yaml và CLI predict.
- Bàn giao theo docs/real_data_workflow.md. Chưa hoàn thành dữ liệu/nhãn/train/test thật thì không gọi hoàn thành thực nghiệm.
