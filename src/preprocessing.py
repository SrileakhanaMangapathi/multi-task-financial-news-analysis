# preprocessing.py
# Handles: cleaning, SMOTE, stratified splits, median imputation

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def clean_headline(text: str) -> str:
    """Lowercase, remove special chars, extra spaces."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['headline_clean'] = df['headline'].apply(clean_headline)
    # Median imputation for numerical features
    for col in ['Index_Change_Percent', 'Trading_Volume']:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    return df

def stratified_split(df, label_col, test_size=0.15, val_size=0.15, seed=42):
    train_val, test = train_test_split(df, test_size=test_size,
                                       stratify=df[label_col], random_state=seed)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio,
                                  stratify=train_val[label_col], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
