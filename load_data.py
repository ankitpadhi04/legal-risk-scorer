from datasets import load_dataset
dataset = load_dataset("theatticusproject/cuad")

print(dataset)
print(dataset["train"].features)
