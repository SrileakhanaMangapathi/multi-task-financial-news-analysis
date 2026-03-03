# Multi-Task Financial News Analysis

A deep learning system that simultaneously performs **Market Event Classification**, **Sentiment Analysis**, and **Impact Level Prediction** from financial news headlines.

## Project Overview

This project builds three complementary models to analyze financial news:

| Model | Architecture | Tasks |
|-------|-------------|-------|
| Model 1 | DistilBERT (fine-tuned) | Market Event + Sentiment |
| Model 2 | BiGRU + FastText | Impact Level Prediction |
| Model 3 | BiLSTM + FastText | Market Event + Sentiment |

## Dataset

[Financial News Market Events Dataset for NLP 2025](https://www.kaggle.com/) — contains financial news headlines with gold labels for:
- **Market Event**: Commodity Price Shock, Central Bank Meeting, etc.
- **Sentiment**: Positive / Negative / Neutral
- **Impact Level**: High / Medium / Low

## Models

### Model 1 — DistilBERT Multi-Task
- Fine-tuned DistilBERT with two classification heads (Market Event + Sentiment)
- Discriminative learning rates (lower for base layers, higher for heads)
- Task-specific loss weighting for balanced optimization

### Model 2 — BiGRU + FastText
- BiGRU with FastText embeddings for Impact Level prediction
- Attention mechanisms: Bahdanau attention, self-attention pooling
- Feature fusion with numerical inputs (`Index_Change_Percent`, `Trading_Volume`)

### Model 3 — BiLSTM Multi-Task
- Bidirectional LSTM with shared encoder + task-specific attention heads
- Attention variants: additive, multiplicative, multi-head self-attention
- Compared against transformer-based approach (Model 1)

## Project Structure
```
├── data/
│   └── raw/                  # Raw dataset files
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_distilbert_multitask.ipynb
│   ├── 03_bigru_impact.ipynb
│   └── 04_bilstm_multitask.ipynb
├── src/
│   ├── preprocessing.py      # Cleaning, SMOTE, stratified splits
│   ├── model_distilbert.py
│   ├── model_bigru.py
│   └── model_bilstm.py
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

To download the dataset via Kaggle API:
```bash
kaggle datasets download -d <dataset-slug>
unzip *.zip -d data/raw/
```

## Preprocessing
- Headline cleaning & normalization
- Missing value imputation (median)
- Class imbalance handling with **SMOTE**
- Stratified train/val/test splits: **70 / 15 / 15**

## Training
- Early stopping
- Learning rate scheduling
- fp16 mixed precision (via `accelerate`)
- Google Colab GPU (T4 / A100)

## Evaluation Metrics
- Macro & Weighted F1
- Per-class Precision / Recall / F1
- Confusion Matrix
- Cohen's Kappa & Matthews Correlation Coefficient
- McNemar's Test for model comparison
- 5-Fold Cross-Validation (mean ± std)
- Attention weight visualization

## Requirements
See [`requirements.txt`](requirements.txt)
