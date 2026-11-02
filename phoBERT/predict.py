
import os
from pathlib import Path

# ==========================================
# 1. CONFIG JAVA
# ==========================================

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.31.11-hotspot"

os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = (
    os.path.join(JAVA_HOME, "bin")
    + os.pathsep
    + os.environ.get("PATH", "")
)

# Import sau khi cấu hình Java
import torch
import py_vncorenlp

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# 2. PATH CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phobert"
    / "best_model"
)

VNCORENLP_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "vncorenlp"
)

MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 3. LOAD TOKENIZER AND MODEL
# ==========================================

print("Loading PhoBERT model...")

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
# 4. INITIALIZE VnCoreNLP
# ==========================================

print("\nInitializing VnCoreNLP...")

segmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"],
    save_dir=str(VNCORENLP_DIR.resolve())
)

print("VnCoreNLP initialized successfully!")


# ==========================================
# 5. LABEL MAPPING
# ==========================================

ID2LABEL = {
    int(key): value
    for key, value in model.config.id2label.items()
}

# ==========================================
# 6. PREDICTION FUNCTION
# ==========================================

def predict_sentiment(text):

    # Word segmentation
    segmented_text = " ".join(
        segmenter.word_segment(text)
    )

    # Tokenization
    encoding = tokenizer(
        segmented_text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    # Inference
    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_id = torch.argmax(
            probabilities
        ).item()

    predicted_label = ID2LABEL[predicted_id]

    confidence = probabilities[predicted_id].item()

    probability_dict = {
        ID2LABEL[i]: probabilities[i].item()
        for i in range(len(probabilities))
    }

    return {
        "original_text": text,
        "segmented_text": segmented_text,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probability_dict
    }


# ==========================================
# 7. INTERACTIVE TERMINAL
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("       PTIT SENTIMENT ANALYSIS - PHOBERT")
    print("=" * 55)

    while True:

        text = input(
            "\nEnter a comment (type 'exit' to quit): "
        ).strip()

        if text.lower() == "exit":
            print("Program terminated!")
            break

        if not text:
            print("Please enter a comment!")
            continue

        try:

            result = predict_sentiment(text)

            print("\nOriginal text:")
            print(result["original_text"])

            print("\nSegmented text:")
            print(result["segmented_text"])

            print("\nPredicted sentiment:")
            print(result["predicted_label"].upper())

            print(
                "\nConfidence: {:.2f}%".format(
                    result["confidence"] * 100
                )
            )

            print("\nProbability distribution:")

            for label, probability in result["probabilities"].items():

                print(
                    f"{label:10s}: {probability * 100:.2f}%"
                )

            print("-" * 55)

        except Exception as error:

            print("Prediction error:", error)

# Hệ thống đăng ký tín chỉ quá tệ, lúc nào cũng bị lỗi.
#Môn học này hay quá, giảng viên dạy rất dễ hiểu.
# Mọi người cho mình hỏi lịch đăng ký học phần khi nào bắt đầu?