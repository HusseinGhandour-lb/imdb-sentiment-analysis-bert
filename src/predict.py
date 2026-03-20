import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained("./results/model")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model.to(device)
model.eval()

text = input("Enter a review: ")

inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

label = "Positive" if prediction == 1 else "Negative"
print(f"Prediction: {label}")