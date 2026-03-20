from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

dataset = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("./results/model")

trainer = Trainer(model=model)

results = trainer.evaluate(tokenized_datasets["test"])
print(results)