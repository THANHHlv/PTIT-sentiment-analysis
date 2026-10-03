"""Định dạng chung cho đánh giá classical, baseline và PhoBERT."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ptit_sentiment.common import write_json
from ptit_sentiment.labels import LABELS


def compute_metrics(true_labels, predicted_labels):
    """Tính đầy đủ metric với cùng thứ tự nhãn; undefined metric -> 0."""
    if not len(true_labels) or len(true_labels) != len(predicted_labels):
        raise ValueError("Danh sách nhãn phải không rỗng và có cùng độ dài.")
    if set(true_labels) - set(LABELS) or set(predicted_labels) - set(LABELS):
        raise ValueError("Metric chỉ nhận ba nhãn trong labels.py.")
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=list(LABELS), zero_division=0)
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "n_samples": len(true_labels),
        "per_label": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate(LABELS)},
        "confusion_matrix": {"labels": list(LABELS), "rows": "true_label",
                             "columns": "predicted_label",
                             "values": confusion_matrix(true_labels, predicted_labels, labels=list(LABELS)).tolist()},
    }


def prediction_frame(frame, predicted, model_name, probabilities=None, classes=None):
    """Giữ id/text gốc; nhãn thật rỗng cho bình luận mới không gán nhãn."""
    result = frame.loc[:, ["id", "text"]].copy()
    result["true_label"] = frame["label"] if "label" in frame.columns else ""
    result["predicted_label"] = list(predicted)
    result["model_name"] = model_name
    if probabilities is not None:
        classes = list(classes)
        probabilities = np.asarray(probabilities)
        if set(classes) != set(LABELS) or probabilities.shape != (len(frame), len(classes)):
            raise ValueError("Xác suất không khớp ba nhãn hoặc số mẫu.")
        for label in LABELS:
            result[f"probability_{label}"] = probabilities[:, classes.index(label)]
    return result


def export_evaluation(output_dir, predictions, metadata):
    """Xuất JSON, CSV tổng/per-label, confusion matrix PNG/CSV, câu sai."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} đã có kết quả; hãy chọn thư mục đánh giá mới.")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(predictions["true_label"].tolist(), predictions["predicted_label"].tolist())
    metrics.update(metadata)
    write_json(output_dir / "metrics.json", metrics)
    summary_keys = ("model_name", "split", "provenance", "split_manifest_sha256",
                    "evaluation_data_sha256", "n_samples", "accuracy",
                    "macro_precision", "macro_recall", "macro_f1")
    pd.DataFrame([{key: metrics[key] for key in summary_keys}]).to_csv(
        output_dir / "metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame.from_dict(metrics["per_label"], orient="index").rename_axis("label").to_csv(
        output_dir / "per_label.csv", encoding="utf-8")
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
    predictions.loc[predictions["true_label"] != predictions["predicted_label"]].to_csv(
        output_dir / "errors.csv", index=False, encoding="utf-8")
    matrix = np.array(metrics["confusion_matrix"]["values"])
    pd.DataFrame(matrix, index=pd.Index(LABELS, name="true_label"),
                 columns=pd.Index(LABELS, name="predicted_label")).to_csv(output_dir / "confusion_matrix.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=axis, colorbar=False, cmap="Blues")
    axis.set_xlabel("Predicted label (nhãn dự đoán)")
    axis.set_ylabel("True label (nhãn thật)")
    axis.set_title(f"{metadata['model_name']} / {metadata['split']}")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    return metrics
