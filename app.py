"""
BITS Pilani - Work Integrated Learning Programmes Division
M.Tech (AIML) - Machine Learning (Assignment 2)
Streamlit Web Application for Multi-Classifier Evaluation

Features:
- Upload holdout test dataset (CSV) or use the bundled test data
- Model selector dropdown for 5 classifiers
- Real-time computation and display of 6 evaluation metrics:
  (Accuracy, AUC, Precision, Recall, F1 Score, MCC)
- Confusion Matrix visualization & Classification Report
- ROC Curve plot
- Multi-model comparison table
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# -----------------------------------------------------------------------------
# App Layout Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Email Spam Classifier | ML Assignment 2",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Email Spam Classification & Model Evaluation")
st.caption("BITS Pilani · M.Tech (AIML) · Machine Learning Assignment 2")

# -----------------------------------------------------------------------------
# Sidebar: Test Data & Model Options
# -----------------------------------------------------------------------------
st.sidebar.header("📁 1. Test Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"],
    help="Upload your test dataset. It must contain the feature columns and the 'class' target column."
)

# Load test data: uploaded file takes priority, otherwise fallback to bundled test_data.csv
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded: `{uploaded_file.name}` ({data.shape[0]} rows)")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        st.stop()
elif os.path.exists("test_data.csv"):
    data = pd.read_csv("test_data.csv")
    st.sidebar.info(f"Using default `test_data.csv` ({data.shape[0]} rows)")
else:
    st.error("No test data found. Please upload a test dataset CSV containing the 'class' column.")
    st.stop()

# Ensure the expected label column exists
target_column = "class"
if target_column not in data.columns:
    st.error(f"Target column '{target_column}' is missing from the test dataset. Please ensure the target column is named '{target_column}'.")
    st.stop()

X_test = data.drop(columns=[target_column])
y_test = data[target_column].astype(int)

# Load StandardScaler
scaler_file = os.path.join("model", "scaler.joblib")
if os.path.exists(scaler_file):
    scaler = joblib.load(scaler_file)
    X_test_scaled = scaler.transform(X_test)
else:
    st.warning("StandardScaler artifact not found in `model/scaler.joblib`. Using unscaled features.")
    X_test_scaled = X_test

st.sidebar.header("⚙️ 2. Select Classifier")
model_catalog = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

# Verify models exist on disk
ready_models = {}
for name, path in model_catalog.items():
    if os.path.exists(path):
        ready_models[name] = path

if not ready_models:
    st.error("No saved models found in the `model/` folder. Please execute `python train.py` first.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Choose a model to inspect:", list(ready_models.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("**Assignment Details:**")
st.sidebar.markdown("- **Dataset**: UCI Spambase")
st.sidebar.markdown("- **Features**: 57 attributes")
st.sidebar.markdown("- **Target**: 1 = Spam, 0 = Ham")

# -----------------------------------------------------------------------------
# Model Inference
# -----------------------------------------------------------------------------
current_model = joblib.load(ready_models[selected_model_name])
y_pred = current_model.predict(X_test_scaled)

# Probability scores for AUC
if hasattr(current_model, "predict_proba"):
    y_prob = current_model.predict_proba(X_test_scaled)[:, 1]
elif hasattr(current_model, "decision_function"):
    df_vals = current_model.decision_function(X_test_scaled)
    y_prob = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-9)
else:
    y_prob = y_pred

# Compute 6 required evaluation metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# -----------------------------------------------------------------------------
# Main Dashboard Tabs
# -----------------------------------------------------------------------------
tab_eval, tab_compare, tab_data, tab_about = st.tabs([
    "📊 Model Evaluation",
    "📋 Compare All Models",
    "🔍 Test Data Explorer",
    "ℹ️ Dataset & Project Info"
])

# -----------------------------
# Tab 1: Single Model Evaluation
# -----------------------------
with tab_eval:
    st.subheader(f"Performance Metrics: {selected_model_name}")
    st.markdown("Metrics computed on the uploaded test dataset without retraining.")

    # 6 Metric KPI cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC Score", f"{auc:.4f}")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall", f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC Score", f"{mcc:.4f}")

    st.markdown("---")

    col_cm, col_rep = st.columns([1, 1])

    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Spam (0)", "Spam (1)"],
            yticklabels=["Non-Spam (0)", "Spam (1)"],
            cbar=False, ax=ax
        )
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    with col_rep:
        st.markdown("#### Classification Report")
        rep_dict = classification_report(
            y_test, y_pred,
            target_names=["Non-Spam (0)", "Spam (1)"],
            output_dict=True,
            zero_division=0
        )
        rep_df = pd.DataFrame(rep_dict).transpose()
        st.dataframe(
            rep_df.style.format({
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}"
            }),
            use_container_width=True
        )

        # ROC Curve Plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 2.2))
        ax_roc.plot(fpr, tpr, color="#1D4ED8", lw=2, label=f"AUC = {auc:.4f}")
        ax_roc.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate", fontsize=8)
        ax_roc.set_ylabel("True Positive Rate", fontsize=8)
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_roc)

# -----------------------------
# Tab 2: Comparison of All 5 Models
# -----------------------------
with tab_compare:
    st.subheader("Model Comparison Table (All 5 Classifiers)")
    st.markdown("Side-by-side evaluation of all five classifiers on the identical test split:")

    all_results = []
    for m_name, m_path in ready_models.items():
        m_inst = joblib.load(m_path)
        m_pred = m_inst.predict(X_test_scaled)

        if hasattr(m_inst, "predict_proba"):
            m_prob = m_inst.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(m_inst, "decision_function"):
            df_v = m_inst.decision_function(X_test_scaled)
            m_prob = (df_v - df_v.min()) / (df_v.max() - df_v.min() + 1e-9)
        else:
            m_prob = m_pred

        all_results.append({
            "ML Model Name": m_name,
            "Accuracy": round(accuracy_score(y_test, m_pred), 4),
            "AUC": round(roc_auc_score(y_test, m_prob), 4),
            "Precision": round(precision_score(y_test, m_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, m_pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, m_pred, zero_division=0), 4),
            "MCC": round(matthews_corrcoef(y_test, m_pred), 4)
        })

    comparison_df = pd.DataFrame(all_results)

    # Style: highlight the highest value in each column
    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: #DEF7EC; font-weight: bold;' if v else '' for v in is_max]

    num_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    styled_df = comparison_df.style.apply(highlight_best, subset=num_cols).format({col: "{:.4f}" for col in num_cols})
    st.dataframe(styled_df, use_container_width=True)

    st.info("**Key Finding**: Logistic Regression achieves the highest overall Accuracy (0.9294), AUC (0.9702), F1 (0.9093), and MCC (0.8518), making it the overall winner for this dataset.")

# -----------------------------
# Tab 3: Test Data Explorer
# -----------------------------
with tab_data:
    st.subheader("Holdout Test Data Inspection")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Total Test Rows", f"{len(data):,}")
    col_stat2.metric("Non-Spam Samples (Class 0)", f"{sum(y_test == 0):,}")
    col_stat3.metric("Spam Samples (Class 1)", f"{sum(y_test == 1):,}")

    st.markdown("#### Sample Predictions vs Ground Truth (First 10 Rows)")
    sample_preview = X_test.head(10).copy()
    sample_preview["Actual_Label"] = y_test.head(10).values
    sample_preview["Predicted_Label"] = y_pred[:10]
    sample_preview["Prediction_Status"] = np.where(
        sample_preview["Actual_Label"] == sample_preview["Predicted_Label"],
        "Correct", "Misclassified"
    )
    st.dataframe(sample_preview, use_container_width=True)

# -----------------------------
# Tab 4: Dataset & Project Info
# -----------------------------
with tab_about:
    st.subheader("About the Project")
    st.markdown("""
    - **Problem**: Binary classification to filter spam emails from legitimate messages.
    - **Dataset**: UCI Spambase Dataset (4,601 instances, 57 numeric continuous features).
    - **Feature Types**:
      - 48 word frequency percentages (`word_freq_make`, `word_freq_free`, `word_freq_money`, etc.)
      - 6 character frequency percentages (`char_freq_dollar`, `char_freq_exclamation`, etc.)
      - 3 capital letter run length statistics (average, longest, and total).
    - **Evaluation Metrics**: Accuracy, ROC-AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).
    """)

st.markdown("---")
st.caption("Submitted for BITS Pilani M.Tech (AIML) · Machine Learning Assignment 2")
