import json
import pandas as pd
import random

random.seed(42)

print("Loading CUAD_v1.json...")
with open("Data/CUAD_v1.json", "r") as f:
    data = json.load(f)

rows = []

for contract in data["data"]:
    title = contract["title"]
    for para in contract["paragraphs"]:
        context = para["context"].strip()
        for qa in para["qas"]:
            question = qa["question"]
            is_negative = qa["is_impossible"] or len(qa["answers"]) == 0

            # Skip very short contexts
            if len(context.split()) < 20:
                continue

            # Combine question + context — BERT [SEP] style
            input_text = question + " [SEP] " + context[:800]

            if not is_negative:
                rows.append({
                    "title": title,
                    "clause_text": input_text,
                    "label": 1
                })
            else:
                rows.append({
                    "title": title,
                    "clause_text": input_text,
                    "label": 0
                })

df = pd.DataFrame(rows)

print(f"\nRaw shape: {df.shape}")
print(f"\nLabel distribution before balancing:")
print(df["label"].value_counts())

# Balance — undersample negatives to match positives
positives = df[df["label"] == 1]
negatives = df[df["label"] == 0].sample(len(positives), random_state=42)
df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal shape: {df.shape}")
print(f"\nLabel distribution after balancing:")
print(df["label"].value_counts())

print(f"\nSample input text:")
print(df["clause_text"].iloc[0][:300])

df.to_csv("Data/clauses_labeled.csv", index=False)
print("\nSaved to Data/clauses_labeled.csv")