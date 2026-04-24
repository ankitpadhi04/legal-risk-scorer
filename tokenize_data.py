import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

# Load cleaned data
df = pd.read_csv("Data/clauses_labeled.csv")
print(f"Loaded {len(df)} rows")

# Drop nulls just in case
df = df.dropna(subset=["clause_text", "label"])
df["label"] = df["label"].astype(int)
print(f"After null drop: {len(df)} rows")

# Load Legal-BERT tokenizer
print("\nLoading Legal-BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["clause_text"],
        truncation=True,        # cut off anything beyond 512 tokens
        padding="max_length",   # pad shorter texts to 512
        max_length=512,         # Legal-BERT's maximum
    )

# Convert pandas DataFrame to HuggingFace Dataset
print("Converting to HuggingFace Dataset...")
hf_dataset = Dataset.from_pandas(df[["clause_text", "label"]])

# Apply tokenization
print("Tokenizing...")
tokenized = hf_dataset.map(tokenize, batched=True, batch_size=64)

# Set format for PyTorch
tokenized = tokenized.with_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

# Train / validation split — 80% train, 20% val
split = tokenized.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

print(f"\nTrain size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")

# Peek at one tokenized example
example = train_dataset[0]
print(f"\nInput IDs shape:      {example['input_ids'].shape}")
print(f"Attention mask shape: {example['attention_mask'].shape}")
print(f"Label:                {example['label']}")

# Save tokenized datasets to disk
print("\nSaving tokenized datasets...")
train_dataset.save_to_disk("Data/train_dataset")
val_dataset.save_to_disk("Data/val_dataset")
print("Saved to Data/train_dataset and Data/val_dataset")
