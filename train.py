import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import os

# ── 1. Verify GPU ─────────────────────────────────────────────────────────────
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2. Load Tokenized Datasets ────────────────────────────────────────────────
print("\nLoading tokenized datasets...")
train_dataset = load_from_disk("Data/train_dataset")
val_dataset   = load_from_disk("Data/val_dataset")

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ── 3. Load Legal-BERT With Classification Head ───────────────────────────────
print("\nLoading Legal-BERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=2                # binary: 0 = Low Risk, 1 = High Risk
)
model.to(device)

# ── 4. Metrics Function ───────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ── 5. Training Arguments ─────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

training_args = TrainingArguments(
    output_dir="model/checkpoints",
    num_train_epochs=4,
    per_device_train_batch_size=16,     # fits comfortably in 6GB with fp16
    per_device_eval_batch_size=32,
    warmup_steps=200,                   # gradual LR warmup
    weight_decay=0.01,                  # regularization
    learning_rate=2e-5,                 # standard for BERT fine-tuning
    fp16=True,                          # half precision — cuts VRAM in half
    eval_strategy="epoch",              # evaluate after every epoch
    save_strategy="epoch",              # save checkpoint every epoch
    load_best_model_at_end=True,        # keep the best checkpoint
    metric_for_best_model="f1",
    logging_steps=50,
    report_to="none"                    # disable wandb
)

# ── 6. Trainer ────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("\nStarting training...")
print("Watch your GPU usage: open another terminal and run → nvidia-smi -l 1")
trainer.train()

# ── 8. Save Final Model ───────────────────────────────────────────────────────
print("\nSaving final model...")
trainer.save_model("model/legal-risk-scorer")
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
tokenizer.save_pretrained("model/legal-risk-scorer")
print("Model saved to model/legal-risk-scorer")

# ── 9. Final Evaluation ───────────────────────────────────────────────────────
print("\nFinal evaluation on validation set...")
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["Low Risk", "High Risk"]))