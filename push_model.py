from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "model/legal-risk-scorer"

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Pushing to HuggingFace Hub...")
model.push_to_hub("ankitpadhi04/legal-risk-scorer")
tokenizer.push_to_hub("ankitpadhi04/legal-risk-scorer")

print("Done! Model live at huggingface.co/ankitpadhi04/legal-risk-scorer")