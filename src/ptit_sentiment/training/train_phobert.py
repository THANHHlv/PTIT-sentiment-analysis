"""Fine-tune PhoBERT tùy chọn; cấu hình smoke riêng không thay thực nghiệm đầy đủ."""
import argparse
from pathlib import Path
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

import numpy as np

from ptit_sentiment.common import environment_versions, load_config, run_cli, write_json
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.evaluation.metrics import compute_metrics
from ptit_sentiment.labels import ID_TO_LABEL, LABEL_TO_ID
from ptit_sentiment.preprocessing.tokenize import PhoBERTPreprocessor
from ptit_sentiment.preprocessing.phobert_input import encode_with_statistics


def train_phobert(splits_dir, config_path, artifact_dir):
    """Train bằng train, chọn epoch bằng Macro-F1 validation; không đánh giá test."""
    config = load_config(config_path)
    max_length = config["max_length"]
    if not 4 <= max_length <= 256:
        raise ValueError("PhoBERT base/large yêu cầu max_length trong [4,256], gồm special tokens.")
    frames, manifest, split_hash = load_split_set(splits_dir)
    artifact_dir = Path(artifact_dir).resolve()
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise ValueError("Thư mục artifact đã có dữ liệu; chọn thư mục PhoBERT mới.")
    try:
        import os
        os.environ['USE_TF'] = '0'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
        import accelerate  # Kiểm tra optional dependency trước khi tải checkpoint.
    except ImportError as error:
        raise ImportError('Cài PhoBERT riêng: pip install -e ".[phobert]"') from error
    set_seed(config["seed"])
    preprocessor = PhoBERTPreprocessor(config["segmenter_dir"], config.get("mask_sensitive", True))
    tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"], revision=config["revision"], use_fast=False,
                                               local_files_only=config.get("local_files_only", False))
    tokenizer.truncation_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        config["checkpoint"], revision=config["revision"], num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL, label2id=LABEL_TO_ID,
        local_files_only=config.get("local_files_only", False))

    class EncodedComments(torch.utils.data.Dataset):
        """Tokenized split local, padding động bằng collator."""
        def __init__(self, frame):
            texts = preprocessor.transform(frame["text"].tolist())
            self.encodings, self.truncation = encode_with_statistics(tokenizer, texts, max_length)
            for record in self.truncation["truncated_samples"]:
                record["id"] = frame["id"].iloc[record["index"]]
            self.labels = [LABEL_TO_ID[label] for label in frame["label"]]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: values[index] for key, values in self.encodings.items()}
            item["labels"] = self.labels[index]
            return item

    def validation_metrics(prediction):
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted = [ID_TO_LABEL[int(index)] for index in np.argmax(logits, axis=-1)]
        truth = [ID_TO_LABEL[int(index)] for index in prediction.label_ids]
        metrics = compute_metrics(truth, predicted)
        return {key: metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}

    train_dataset = EncodedComments(frames["train"])
    validation_dataset = EncodedComments(frames["validation"])
    args = TrainingArguments(
        output_dir=str(artifact_dir / "checkpoints"), eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=2, seed=config["seed"], data_seed=config["seed"],
        report_to="none", **config["training"])
    print(f"Thiết bị huấn luyện: {args.device}; checkpoint: {config['checkpoint']}")
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                      eval_dataset=validation_dataset, tokenizer=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=validation_metrics)
    write_json(artifact_dir / "truncation.json",
               {"train": train_dataset.truncation, "validation": validation_dataset.truncation})
    trainer.train()
    reference = trainer.predict(validation_dataset).predictions
    if isinstance(reference, tuple):
        reference = reference[0]
    write_json(artifact_dir / "validation_reference.json", {
        "ids": frames["validation"]["id"].tolist(),
        "predicted_labels": [ID_TO_LABEL[int(index)] for index in np.argmax(reference, axis=-1)],
        "logits": reference.tolist(),
    })
    trainer.save_model(str(artifact_dir))
    tokenizer.save_pretrained(str(artifact_dir))
    trainer.save_state()
    write_json(artifact_dir / "labels.json", {"label_to_id": LABEL_TO_ID})
    saved_config = {**config, "segmenter_dir": str(Path(config["segmenter_dir"]).resolve())}
    write_json(artifact_dir / "training_config.json", saved_config)
    write_json(artifact_dir / "metadata.json", {
        "kind": "phobert", "model_name": "phobert", "label_to_id": LABEL_TO_ID,
        "split_manifest_sha256": split_hash,
        "split_file_hashes": {name: info["sha256"] for name, info in manifest["splits"].items()},
        "provenance": manifest["provenance"], "config": saved_config,
        "versions": environment_versions(), "fit_split": "train", "device": str(args.device),
        "selection_metric": "validation_macro_f1", "best_validation_macro_f1": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "resolved_checkpoint_commit": getattr(model.config, "_commit_hash", None),
    })
    print(f"Đã lưu checkpoint tốt nhất theo validation: {artifact_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--config", default="configs/phobert.yaml")
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    train_phobert(args.splits_dir, args.config, args.artifact_dir)


if __name__ == "__main__":
    run_cli(main)
