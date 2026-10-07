"""PhoBERT tùy chọn; chỉ import torch/transformers khi tải artifact."""
from pathlib import Path
import sys
from ptit_sentiment.preprocessing.phobert_input import encode_with_statistics

from ptit_sentiment.common import read_json
from ptit_sentiment.labels import ID_TO_LABEL, LABELS, LABEL_TO_ID
from ptit_sentiment.preprocessing.tokenize import PhoBERTPreprocessor


class PhoBERTPredictor:
    """Dự đoán từ artifact local; không tải checkpoint và không huấn luyện lại."""

    def __init__(self, artifact_dir, segmenter_dir=None):
        directory = Path(artifact_dir)
        self.metadata = read_json(directory / "metadata.json")
        if self.metadata.get("label_to_id") != LABEL_TO_ID:
            raise ValueError("Ánh xạ nhãn artifact PhoBERT khác labels.py.")
        try:
            import os
            os.environ['USE_TF'] = '0'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ImportError('PhoBERT cần pip install -e ".[phobert]"') from error
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=False, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True)
        if self.model.config.label2id != LABEL_TO_ID:
            raise ValueError("label2id trong model config khác hợp đồng.")
        use_cpu = self.metadata["config"].get("training", {}).get("use_cpu", False)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
        print(f"Thiết bị PhoBERT: {self.device}", file=sys.stderr)
        self.model.to(self.device).eval()
        self.preprocessor = PhoBERTPreprocessor(segmenter_dir or self.metadata["config"]["segmenter_dir"],
                                                 self.metadata["config"].get("mask_sensitive", True))
        self.max_length = self.metadata["config"]["max_length"]

    def predict(self, texts, batch_size=16):
        """Trả nhãn, softmax probability và thứ tự cột theo labels.py."""
        if batch_size < 1:
            raise ValueError("batch_size phải >= 1.")
        if not texts:
            raise ValueError("Danh sách bình luận không được rỗng.")
        probabilities = []
        self.tokenizer.truncation_side = "right"
        self.truncation_stats = {"samples": len(texts), "max_length": self.max_length,
                                 "truncation_side": "right", "truncated_samples": []}
        for start in range(0, len(texts), batch_size):
            segmented = self.preprocessor.transform(texts[start:start + batch_size])
            inputs, stats = encode_with_statistics(self.tokenizer, segmented, self.max_length,
                                                    padding=True, return_tensors="pt")
            self.truncation_stats["truncated_samples"].extend(
                {**record, "index": start + record["index"]} for record in stats["truncated_samples"])
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities.extend(self.torch.softmax(logits, dim=-1).cpu().numpy())
        self.truncation_stats["truncated_count"] = len(self.truncation_stats["truncated_samples"])
        self.truncation_stats["truncated_fraction"] = self.truncation_stats["truncated_count"] / len(texts)
        import numpy as np
        probabilities = np.asarray(probabilities)
        labels = [ID_TO_LABEL[int(index)] for index in probabilities.argmax(axis=1)]
        return labels, probabilities, LABELS
