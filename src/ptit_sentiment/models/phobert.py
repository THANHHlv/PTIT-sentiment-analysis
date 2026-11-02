
"""
PhoBERT Predictor.

Load checkpoint PhoBERT đã fine-tune và dự đoán cảm xúc.
Thứ tự nhãn đầu ra luôn thống nhất với ptit_sentiment.labels.
"""

import sys
from pathlib import Path

from ptit_sentiment.common import read_json
from ptit_sentiment.labels import (
    ID_TO_LABEL,
    LABELS,
    LABEL_TO_ID,
)
from ptit_sentiment.preprocessing.phobert_input import (
    encode_with_statistics,
)
from ptit_sentiment.preprocessing.tokenize import PhoBERTPreprocessor


class PhoBERTPredictor:
    """
    PhoBERT predictor.

    Input:
        texts: danh sách bình luận tiếng Việt.

    Output:
        labels: nhãn dự đoán.
        probabilities: xác suất theo thứ tự LABELS.
        LABELS: thứ tự nhãn chung của dự án.
    """

    def __init__(self, artifact_dir, segmenter_dir=None):

        # =====================================
        # 1. Kiểm tra đường dẫn artifact
        # =====================================

        self.artifact_dir = Path(artifact_dir)

        if not self.artifact_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy artifact: {self.artifact_dir}"
            )

        config_path = self.artifact_dir / "training_config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy training_config.json: {config_path}"
            )

        self.config = read_json(config_path)

        # =====================================
        # 2. Đọc mapping nhãn checkpoint
        # =====================================

        checkpoint_label2id = {
            str(label): int(index)
            for label, index in self.config["label2id"].items()
        }

        # Kiểm tra tập nhãn có giống dự án không.
        if set(checkpoint_label2id.keys()) != set(LABEL_TO_ID.keys()):
            raise ValueError(
                "Các nhãn trong checkpoint không khớp labels.py.\n"
                f"Checkpoint: {checkpoint_label2id}\n"
                f"Project: {LABEL_TO_ID}"
            )

        # Lưu thứ tự index tương ứng với LABELS của dự án.
        #
        # Ví dụ:
        # Project:   positive, neutral, negative
        # Checkpoint: negative, neutral, positive
        #
        # output_indices = [2, 1, 0]

        self.output_indices = [
            checkpoint_label2id[label]
            for label in LABELS
        ]

        # =====================================
        # 3. Import PyTorch và Transformers
        # =====================================

        try:
            import os

            os.environ["USE_TF"] = "0"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

            import torch

            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

        except ImportError as error:
            raise ImportError(
                'PhoBERT cần cài dependencies: pip install -e ".[phobert]"'
            ) from error

        self.torch = torch

        # =====================================
        # 4. Load tokenizer
        # =====================================

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.artifact_dir,
            use_fast=False,
            local_files_only=True,
        )

        # =====================================
        # 5. Load model
        # =====================================

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.artifact_dir,
            local_files_only=True,
        )

        # Kiểm tra mapping trong config.json của model.
        model_label2id = {
            str(label): int(index)
            for label, index in self.model.config.label2id.items()
        }

        if set(model_label2id.keys()) != set(LABEL_TO_ID.keys()):
            raise ValueError(
                "Các nhãn trong model config không khớp labels.py.\n"
                f"Model: {model_label2id}\n"
                f"Project: {LABEL_TO_ID}"
            )

        # Kiểm tra mapping trong config model và training config
        if model_label2id != checkpoint_label2id:
            raise ValueError(
                "Mapping nhãn giữa config.json và training_config.json "
                "không giống nhau."
            )

        # =====================================
        # 6. Chọn thiết bị
        # =====================================

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(
            f"Thiết bị PhoBERT: {self.device}",
            file=sys.stderr,
        )

        self.model.to(self.device)
        self.model.eval()

        # =====================================
        # 7. Khởi tạo Vietnamese Preprocessor
        # =====================================

        if segmenter_dir is None:
            segmenter_dir = self.config.get("segmenter_dir")

        if segmenter_dir is None:
            raise ValueError(
                "Không tìm thấy segmenter_dir. "
                "Hãy truyền đường dẫn VnCoreNLP."
            )

        self.preprocessor = PhoBERTPreprocessor(
            segmenter_dir,
            self.config.get("mask_sensitive", True),
        )

        self.max_length = int(self.config["max_length"])

        print(
            "PhoBERT loaded successfully!",
            file=sys.stderr,
        )

        print(
            f"Thứ tự nhãn đầu ra: {LABELS}",
            file=sys.stderr,
        )

    # =========================================
    # 8. Prediction
    # =========================================

    def predict(self, texts, batch_size=16):
        """
        Dự đoán cảm xúc.

        Args:
            texts: list[str] hoặc str.
            batch_size: số lượng bình luận mỗi batch.

        Returns:
            labels: list nhãn dự đoán.
            probabilities: numpy array xác suất.
            LABELS: thứ tự nhãn.
        """

        import numpy as np

        # Cho phép truyền một chuỗi đơn.
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise ValueError(
                "Danh sách bình luận không được rỗng."
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size phải >= 1."
            )

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Mọi bình luận phải là chuỗi không rỗng."
            )

        probabilities = []

        self.tokenizer.truncation_side = "right"

        self.truncation_stats = {
            "samples": len(texts),
            "max_length": self.max_length,
            "truncation_side": "right",
            "truncated_samples": [],
        }

        # =====================================
        # 9. Xử lý từng batch
        # =====================================

        for start in range(0, len(texts), batch_size):

            batch_texts = texts[start:start + batch_size]

            # Tiền xử lý và phân đoạn từ tiếng Việt.
            segmented = self.preprocessor.transform(
                batch_texts
            )

            # Tokenization.
            inputs, stats = encode_with_statistics(
                self.tokenizer,
                segmented,
                self.max_length,
                padding=True,
                return_tensors="pt",
            )

            # Lưu thông tin các mẫu bị truncation.
            self.truncation_stats["truncated_samples"].extend(
                {
                    **record,
                    "index": start + record["index"],
                }
                for record in stats["truncated_samples"]
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            # Model inference.
            with self.torch.no_grad():

                outputs = self.model(**inputs)

                logits = outputs.logits

                batch_probabilities = self.torch.softmax(
                    logits,
                    dim=-1,
                )

                probabilities.extend(
                    batch_probabilities.cpu().numpy()
                )

        # =====================================
        # 10. Tổng hợp xác suất
        # =====================================

        probabilities = np.asarray(probabilities)

        # Thống kê truncation.
        self.truncation_stats["truncated_count"] = len(
            self.truncation_stats["truncated_samples"]
        )

        self.truncation_stats["truncated_fraction"] = (
            self.truncation_stats["truncated_count"]
            / len(texts)
        )

        # =====================================
        # 11. Chuẩn hóa thứ tự xác suất
        # =====================================

        # Xác suất ban đầu theo checkpoint:
        # negative, neutral, positive
        #
        # Xác suất sau khi chuyển đổi:
        # positive, neutral, negative

        probabilities = probabilities[:, self.output_indices]

        # Lấy ID nhãn theo thứ tự LABELS của dự án.
        predicted_indices = probabilities.argmax(axis=1)

        labels = [
            ID_TO_LABEL[int(index)]
            for index in predicted_indices
        ]

        return labels, probabilities, LABELS
