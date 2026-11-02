
import torch

from transformers import AutoModelForSequenceClassification


# ==========================================
# 1. CONFIG
# ==========================================

MODEL_NAME = "vinai/phobert-base"

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# ==========================================
# 2. LOAD MODEL
# ==========================================

print("Loading PhoBERT Classification Model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

print("Model loaded successfully!")


# ==========================================
# 3. CHECK MODEL
# ==========================================

print("\nModel architecture:")
print(model.classifier)

print("\nNumber of parameters:")

total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ==========================================
# 4. TEST FORWARD
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

# Batch giả lập gồm 2 câu, mỗi câu dài 128 token
input_ids = torch.randint(
    low=0,
    high=1000,
    size=(2, 128)
).to(device)

attention_mask = torch.ones(
    (2, 128),
    dtype=torch.long
).to(device)

with torch.no_grad():

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

logits = outputs.logits

print("\n========== FORWARD TEST ==========")

print("Device:", device)

print("Logits shape:", logits.shape)

print("Logits:")
print(logits)

predictions = torch.argmax(logits, dim=1)

print("\nPredicted label IDs:")
print(predictions)
