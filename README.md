# 🎬 IMDB Sentiment Analysis with BERT

This project uses a pre-trained BERT model to perform sentiment analysis on the IMDB movie reviews dataset.

## 🚀 Features
- Uses Hugging Face Transformers
- Fine-tunes BERT (`bert-base-uncased`)
- Binary classification (Positive / Negative)
- Training + Evaluation + Prediction pipeline

## 📂 Project Structure
src/ → training & inference scripts
notebooks/ → original Kaggle notebook
results/ → trained model outputs


## ⚙️ Installation

```bash
git clone https://github.com/HusseinGhandour/imdb-sentiment-bert.git
cd imdb-sentiment-bert
pip install -r requirements.txt
```

### 🏋️ Training
python src/train.py

### 📊 Evaluation
python src/evaluate.py

### 🔍 Prediction
python src/predict.py

## 🧠 Model
- BERT base uncased
- 2 epochs training
- Learning rate: 2e-5

## 📚 Dataset
IMDB dataset from Hugging Face

## ✨ Author
Hussein Ghandour
