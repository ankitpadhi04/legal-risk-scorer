import torch
import shap
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Model ─────────────────────────────────────────────────────────────
print("Loading model...")
model_path = "model/legal-risk-scorer"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ── 2. Define Prediction Function For SHAP ───────────────────────────────────
def predict(texts):
    """Takes a list of strings, returns probability of both classes."""
    # SHAP sometimes passes numpy arrays — convert to strings
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    texts = [str(t) for t in texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).numpy()
    return probs

# ── 3. Test Prediction Works ──────────────────────────────────────────────────
question = 'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer.'
context = "The licensee shall not compete with the licensor in any market for a period of 5 years following termination of this agreement."
test_clause = question + " [SEP] " + context

probs = predict([test_clause])
print(f"\nTest prediction:")
print(f"  Low Risk:  {probs[0][0]:.3f}")
print(f"  High Risk: {probs[0][1]:.3f}")
print(f"  Verdict:   {'High Risk' if probs[0][1] > 0.5 else 'Low Risk'}")

# ── 4. SHAP Explainer ─────────────────────────────────────────────────────────
print("\nInitializing SHAP explainer...")
explainer = shap.Explainer(predict, tokenizer, output_names=["Low Risk", "High Risk"])

# ── 5. Generate SHAP Values ───────────────────────────────────────────────────
print("Generating SHAP values (takes 30-60 seconds)...")
shap_values = explainer([test_clause], fixed_context=1)

# ── 6. Print Token Attributions ──────────────────────────────────────────────
tokens = shap_values.data[0]
values = shap_values.values[0, :, 1]  # index 1 = High Risk class

token_importance = list(zip(tokens, values))
token_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\nTop 10 most influential tokens:")
for token, importance in token_importance[:10]:
    direction = "↑ Risk" if importance > 0 else "↓ Risk"
    print(f"  {token:20s} {importance:+.4f}  {direction}")

print("\nSHAP working correctly!")