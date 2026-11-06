"""
BITS Pilani - Work Integrated Learning Programmes Division
M.Tech (AIML) - Machine Learning (Assignment 2)
Student Implementation: Model Training & Evaluation Pipeline

Dataset: UCI Spambase Dataset (4601 rows, 57 features)
Models:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier (KNN)
  4. Naive Bayes Classifier (GaussianNB)
  5. Support Vector Machine (SVC)
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Suppress minor version warnings for clean terminal display
warnings.filterwarnings("ignore")


def main():
    print("=" * 65)
    print(" BITS Pilani ML Assignment 2 - Model Training Pipeline ")
    print("=" * 65)

    # -------------------------------------------------------------
    # Step 1: Load the Dataset
    # -------------------------------------------------------------
    print("\n[Step 1] Loading UCI Spambase dataset via OpenML...")
    dataset = fetch_openml('spambase', version=1, as_frame=True)
    df = dataset.frame

    X = df.drop(columns=['class'])
    y = df['class'].astype(int)

    print(f" -> Total samples: {X.shape[0]}")
    print(f" -> Feature count: {X.shape[1]} (Meets >12 requirement)")
    print(f" -> Class distribution: Non-Spam={sum(y == 0)}, Spam={sum(y == 1)}")

    # -------------------------------------------------------------
    # Step 2: Train/Test Split (80% train, 20% holdout test)
    # -------------------------------------------------------------
    print("\n[Step 2] Performing stratified 80-20 train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save holdout test data to CSV for Streamlit deployment upload
    test_export = X_test.copy()
    test_export['class'] = y_test.values
    test_export.to_csv("test_data.csv", index=False)
    print(f" -> Saved 'test_data.csv' ({len(test_export)} rows, {test_export.shape[1]} cols) for Streamlit upload.")

    # -------------------------------------------------------------
    # Step 3: Feature Scaling (Fitted only on train to avoid leakage)
    # -------------------------------------------------------------
    print("\n[Step 3] Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.joblib")
    print(" -> Saved preprocessor: model/scaler.joblib")

    # -------------------------------------------------------------
    # Step 4: Define the 5 Classification Models
    # -------------------------------------------------------------
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42)
    }

    # -------------------------------------------------------------
    # Step 5: Train, Evaluate, and Save Models
    # -------------------------------------------------------------
    print("\n[Step 4] Training models and computing evaluation metrics...")
    metric_records = []

    for model_name, model in classifiers.items():
        print(f" -> Training {model_name}...")
        model.fit(X_train_scaled, y_train)

        # Save model artifact
        save_path = os.path.join("model", f"{model_name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, save_path)

        # Predict on holdout test set
        y_pred = model.predict(X_test_scaled)
        
        # Get probability estimates for AUC calculation
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred

        # Compute all 6 required evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        metric_records.append({
            "ML Model Name": model_name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })

    # -------------------------------------------------------------
    # Step 6: Print Evaluation Results Table
    # -------------------------------------------------------------
    summary_df = pd.DataFrame(metric_records)
    print("\n" + "=" * 70)
    print(" MODEL PERFORMANCE COMPARISON TABLE (20% Holdout Test Set) ")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    # Save to disk
    summary_df.to_csv("model/metrics_summary.csv", index=False)
    print("\nAll model artifacts and metrics successfully generated!")


if __name__ == "__main__":
    main()
