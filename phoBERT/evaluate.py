
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from tqdm import tqdm

from dataset import PhoBERTDataset, LABEL2ID, ID2LABEL


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test.csv"
)

MODEL_DIR = (
    PROJECT_ROOT / "outputs" / "phobert" / "best_model"
)

RESULT_DIR = (
    PROJECT_ROOT / "outputs" / "phobert" / "evaluation"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 128
BATCH_SIZE = 2

LABEL_NAMES = [
    ID2LABEL[i] for i in range(len(ID2LABEL))
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_FP16 = DEVICE.type == "cuda"


# ==========================================
# 2. CHECK FILES
# ==========================================

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_DIR}")


# ==========================================
# 3. LOAD TOKENIZER AND MODEL
# ==========================================

print("Loading best PhoBERT model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=False
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print("Device:", DEVICE)


# ==========================================
# 4. LOAD TEST DATASET
# ==========================================

test_dataset = PhoBERTDataset(
    csv_path=TEST_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# ==========================================
# 5. PREDICTION
# ==========================================

all_predictions = []
all_labels = []
all_probabilities = []

print("\nEvaluating on test set...")

with torch.no_grad():

    for batch in tqdm(test_loader, desc="Testing"):

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_FP16):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = outputs.logits

        probabilities = torch.softmax(logits, dim=1)

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ==========================================
# 6. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1, 2],
    target_names=LABEL_NAMES,
    zero_division=0,
    digits=4
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(report)


# ==========================================
# 7. SAVE CLASSIFICATION REPORT
# ==========================================

report_dict = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1, 2],
    target_names=LABEL_NAMES,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = RESULT_DIR / "classification_report.csv"

report_df.to_csv(
    report_path,
    encoding="utf-8-sig"
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2]
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("PhoBERT Confusion Matrix")

plt.tight_layout()

cm_path = RESULT_DIR / "confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 9. SAVE PREDICTION DETAILS
# ==========================================

test_df = pd.read_csv(TEST_PATH)

prediction_df = pd.DataFrame({
    "text": test_df["text"].astype(str),
    "segmented_text": test_df["segmented_text"].astype(str),
    "true_label": [
        ID2LABEL[int(label)] for label in all_labels
    ],
    "predicted_label": [
        ID2LABEL[int(prediction)] for prediction in all_predictions
    ],
    "confidence": np.max(all_probabilities, axis=1)
})

prediction_path = RESULT_DIR / "test_predictions.csv"

prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 10. SAVE SUMMARY
# ==========================================

summary = {
    "model": "PhoBERT",
    "test_samples": len(test_dataset),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "confusion_matrix": cm.tolist()
}

summary_path = RESULT_DIR / "summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        ensure_ascii=False,
        indent=4
    )


# ==========================================
# 11. FINISHED
# ==========================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED!")
print("=" * 60)

print("Classification report:", report_path)
print("Confusion matrix:", cm_path)
print("Predictions:", prediction_path)
print("Summary:", summary_path)
