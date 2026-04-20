# src/models/train_classifier.py
# Run this once to train and save the best model

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.preprocessing.text_cleaner import preprocess_dataframe

SAVE_PATH = 'src/models/saved/best_classifier.pkl'
DATA_PATH = 'data/raw/cust_support_ticket.csv'


def build_pipeline():
    # Use LinearSVC - This is the engine that got you 77%
    # Ensure C=1.5 and class_weight='balanced'
    svc = LinearSVC(C=5, class_weight='balanced', dual=False, max_iter=2000)

    svm_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            norm='l2',
            # Ensure stop_words matches your Colab exactly
            stop_words='english'
        )),
        ('clf', CalibratedClassifierCV(svc, cv=5, method='sigmoid'))
    ])

    # Wrap for confidence scores
    return svm_pipe


def train():
    print("Loading data...")
    df_raw = pd.read_csv(DATA_PATH)

    print("Preprocessing...")
    df = preprocess_dataframe(df_raw)
    print(f"Rows after preprocessing: {len(df)}")
    print(f"Category distribution:\n{df['category'].value_counts()}\n")

    X = df['final_text']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n=== Evaluation on held-out test set ===")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    joblib.dump(pipeline, SAVE_PATH)
    print(f"\nSaved to: {SAVE_PATH}")
    print(f"Classes: {list(pipeline.classes_)}")


if __name__ == '__main__':
    train()