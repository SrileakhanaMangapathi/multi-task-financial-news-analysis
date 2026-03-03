# 📊 Multi-Task Financial News Analysis (NLP)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

------------------------------------------------------------------------

## 🚀 Project Overview

This project implements a **multi-task NLP system** for analyzing
financial news headlines.\
The model jointly predicts:

-   📰 **Market Event Classification**
-   💬 **Sentiment Prediction**
-   📈 **Impact Level Modeling**

The goal is to extract structured market intelligence from unstructured
financial news data using deep learning.

------------------------------------------------------------------------

## 🏗️ Model Architecture

![Architecture Diagram](architecture_diagram.png)

### 🔹 Transformer Branch (Multi-Task)

-   DistilBERT shared encoder
-   Two task-specific heads:
    -   Market Event Classification
    -   Sentiment Classification
-   Weighted multi-task loss

### 🔹 RNN Branch (Impact Modeling)

-   FastText embeddings
-   BiGRU / BiLSTM architecture
-   Optional numeric feature fusion
-   Impact level prediction

------------------------------------------------------------------------

## 🧠 Models Implemented

### 1️⃣ DistilBERT Multi-Task Model

-   Shared transformer backbone
-   Dual classification heads
-   Fine-tuned on financial dataset

### 2️⃣ BiGRU + FastText (Impact Prediction)

-   Bidirectional GRU
-   Pre-trained word embeddings
-   Impact-level classification

### 3️⃣ BiLSTM Multi-Task Baseline

-   Bidirectional LSTM
-   Multi-head event + sentiment prediction

------------------------------------------------------------------------

## 📂 Dataset

**Financial News Market Events Dataset for NLP 2025**

Source:\
https://www.kaggle.com/datasets/pratyushpuri/financial-news-market-events-dataset-2025

Dataset contains \~3,000 synthetic financial news headlines including: -
Market event labels - Sentiment labels - Impact levels - Optional
numeric indicators

⚠️ Dataset not included in repository due to Kaggle licensing.

------------------------------------------------------------------------

## 🏗️ Project Structure

    multi-task-financial-news-analysis/
    │
    ├── src/
    │   ├── train_distilbert_multitask.py
    │   ├── train_bigru_impact.py
    │   ├── train_bilstm_multitask.py
    │   ├── models_distilbert.py
    │   ├── models_rnn.py
    │   ├── data.py
    │   ├── metrics.py
    │   └── evaluate.py
    │
    ├── configs/
    ├── notebooks/
    ├── docs/
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
git clone https://github.com/<your-username>/multi-task-financial-news-analysis.git
cd multi-task-financial-news-analysis

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📥 Dataset Setup

1.  Download dataset from Kaggle.
2.  Place CSV file inside:

```{=html}
<!-- -->
```
    data/raw/financial_news_events.csv

3.  Update column names in `configs/*.yaml` if needed.

------------------------------------------------------------------------

## ▶️ Training

### Train DistilBERT Multi-Task

    python -m src.train_distilbert_multitask --config configs/distilbert_multitask.yaml

### Train BiGRU Impact Model

    python -m src.train_bigru_impact --config configs/bigru_impact.yaml

### Train BiLSTM Multi-Task

    python -m src.train_bilstm_multitask --config configs/bilstm_multitask.yaml

------------------------------------------------------------------------

## 📊 Evaluation Metrics

-   Macro F1 Score\
-   Weighted F1 Score\
-   Confusion Matrix\
-   Matthews Correlation Coefficient (MCC)\
-   Cohen's Kappa

These metrics ensure robust evaluation across class imbalance scenarios.

------------------------------------------------------------------------

## 🔬 Key Highlights

-   Multi-task learning architecture for financial NLP\
-   Transformer vs RNN comparative study\
-   Modular training + evaluation pipeline\
-   Configuration-driven experiments\
-   Production-ready project structure

------------------------------------------------------------------------

## 💡 Future Improvements

-   Add attention mechanisms to RNN models\
-   Hyperparameter tuning\
-   Integrate FinBERT\
-   Add explainability (SHAP / attention visualization)\
-   Deploy as API or Streamlit app
