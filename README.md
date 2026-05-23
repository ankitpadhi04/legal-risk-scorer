# ⚖️ Legal Risk Scorer

> *Fine-tuned Legal-BERT · CUAD Dataset · SHAP Explainability · Streamlit*

A machine learning application that analyzes legal contract clauses and flags potentially risky language — with token-level explanations showing exactly *why* a clause was flagged.

---

## 🎯 What It Does

Paste any contract clause or a full contract and the app will:

- **Classify** it as High Risk or Low Risk with a confidence score
- **Explain** which specific words and phrases drove the prediction using SHAP
- **Scan** full contracts by auto-splitting them into clause-level chunks
- **Summarize** overall contract risk across multiple clause types

---

## 🖥️ Live Demo

> **[Launch App →](https://your-username.streamlit.app/legal-risk-scorer)**

---

## 🧠 Model & Dataset

### Dataset — CUAD (Contract Understanding Atticus Dataset)

The model was trained on the **CUAD dataset** — a benchmark created by legal experts at The Atticus Project consisting of 500+ real commercial contracts with 13,000+ labeled clause instances across 41 clause categories.

| Split | Examples |
|---|---|
| Train | 15,867 |
| Validation | 3,967 |
| Total | 19,834 |

Labels are **balanced 50/50** between High Risk and Low Risk using proper Q&A-style negatives — each negative example uses the same contract paragraph context as the positive, ensuring the model learns genuine legal risk signals rather than surface-level text patterns.

### Model — Legal-BERT

Built on **`nlpaueb/legal-bert-base-uncased`** — a BERT-base model pretrained specifically on legal corpora including contracts, court cases, and legislation. Fine-tuned for binary sequence classification.

| Metric | Score |
|---|---|
| Accuracy | 81% |
| F1 (Low Risk) | 0.81 |
| F1 (High Risk) | 0.82 |
| Parameters | ~110M |
| Max Token Length | 512 |

Training was done locally on an **NVIDIA RTX 3060 (6GB VRAM)** using `fp16` mixed precision, completing in ~45 minutes.

---

## 🔍 Explainability — SHAP

Every prediction comes with a **SHAP token attribution chart** showing which words pushed the risk score up or down.

```
compete      ████████████  ↑ High Risk
licensor     ██████        ↑ High Risk  
shall        ████          ↓ Low Risk
any          ███           ↓ Low Risk
```

Red tokens increase risk probability. Green tokens decrease it. This makes the model's reasoning transparent and auditable — not a black box.

---

## 🏗️ Architecture

```
Raw CUAD PDFs (511 contracts)
        ↓
CUADv1.json — Q&A format (20,910 labeled examples)
        ↓
Preprocessing — question + [SEP] + context → binary label
        ↓
Tokenization — Legal-BERT tokenizer (max 512 tokens)
        ↓
Fine-tuning — Legal-BERT + Classification Head
        ↓
SHAP Explainer — token-level attributions
        ↓
Streamlit App — Single clause + Full contract scan
        ↓
HuggingFace Hub — model weights hosted
        ↓
Streamlit Community Cloud — live deployment
```

---

## 🗂️ Project Structure

```
legal-risk-scorer/
│
├── app.py                  # Streamlit frontend — main application
│
├── preprocess.py           # CUADv1.json → balanced labeled CSV
├── tokenize_data.py        # Legal-BERT tokenization → HF datasets
├── train.py                # Fine-tuning with HuggingFace Trainer API
├── explain.py              # SHAP explainability testing script
├── push_model.py           # Push trained model to HuggingFace Hub
│
├── explore_data.py         # EDA on master_clauses.csv
├── load_data.py            # Raw CUAD PDF exploration
├── verify.py               # GPU + environment verification
│
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages for Streamlit Cloud
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Tool |
|---|---|
| Model Architecture | Legal-BERT (`nlpaueb/legal-bert-base-uncased`) |
| Training Framework | HuggingFace Transformers + Trainer API |
| Explainability | SHAP (Partition Explainer) |
| Frontend | Streamlit |
| Visualization | Plotly |
| Dataset | CUAD via `CUADv1.json` |
| Model Hosting | HuggingFace Hub |
| Deployment | Streamlit Community Cloud |
| Training Hardware | NVIDIA RTX 3060 6GB (fp16) |

---

## 🚀 Run Locally

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x (for training only; inference runs on CPU)

### Setup

```bash
git clone https://github.com/your-username/legal-risk-scorer.git
cd legal-risk-scorer

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Run The App

```bash
streamlit run app.py
```

The app loads the model directly from HuggingFace Hub on first run (~440MB download, cached after).

### Retrain From Scratch (Optional)

Download `CUADv1.json` from [Kaggle — CUAD Dataset](https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta) and place it in a `Data/` folder, then:

```bash
python preprocess.py       # build labeled dataset
python tokenize_data.py    # tokenize with Legal-BERT
python train.py            # fine-tune on GPU (~45 min)
python push_model.py       # push to HuggingFace Hub
```

---

## 🧪 Supported Clause Types

The app checks for 8 high-risk clause categories:

| Clause Type | Why It's Risky |
|---|---|
| Non-Compete | Restricts future employment or business activity |
| Uncapped Liability | Unlimited financial exposure |
| Termination For Convenience | Other party can exit anytime without cause |
| IP Ownership Assignment | Your intellectual property may belong to them |
| Anti-Assignment | Cannot transfer contract to another party |
| Liquidated Damages | Pre-set penalty amounts regardless of actual harm |
| Change Of Control | Contract terms change if company is acquired |
| Auto-Renewal | Contract renews automatically without explicit consent |

---

## 📋 Input Format

**Single Clause Analysis:**
Select the clause type from the dropdown and paste the clause text. The model uses the clause type to frame the question in the same format it was trained on.

**Full Contract Scan:**
Paste the full contract text. The app automatically detects clause boundaries using section markers (`1.`, `Article`, `Section`, `WHEREAS` etc.) and scans each chunk independently.

---

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It does not constitute legal advice. Always consult a qualified attorney before making decisions based on contract analysis. Model predictions may be incorrect — 81% accuracy means approximately 1 in 5 predictions may be wrong.

---

## 👤 Author

**Ankit Padhi**
B.Tech Computer Science & Engineering — KiiT University

---

*Built with Legal-BERT, CUAD, SHAP, and a lot of GPU time.*
