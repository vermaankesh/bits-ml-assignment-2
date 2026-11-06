# Machine Learning Classification Assignment - 2
**BITS Pilani — Work Integrated Learning Programmes Division**  
**Programme**: M.Tech (AIML)  
**Course**: Machine Learning  
**Assignment**: Assignment - 2  

---

## a. Problem Statement
The objective of this project is to develop, evaluate, and deploy an end-to-end supervised machine learning pipeline that classifies incoming emails into **Spam** (unsolicited commercial or phishing emails) or **Non-Spam (Ham)** based on word and character frequency attributes. Identifying spam accurately is vital for cybersecurity, preventing phishing scams, minimizing bandwidth clutter, and ensuring productivity. 

Five distinct classification models are implemented on the same dataset, rigorously evaluated across six standard statistical evaluation metrics, and deployed as an interactive web application on Streamlit Community Cloud without requiring retraining during inference.

---

## b. Dataset Description
- **Dataset Name**: Spambase Dataset (UCI Machine Learning Repository)
- **Source**: Created by Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt (Hewlett-Packard Laboratories).
- **Public Repository Link**: [UCI Spambase Repository](https://archive.ics.uci.edu/dataset/94/spambase) / OpenML ID: 44
- **Total Number of Instances**: **4,601** (Exceeds the minimum threshold of 500 instances)
- **Total Number of Features**: **57 input features** (Exceeds the minimum threshold of 12 features)
- **Target Variable**: `class` (Binary classification: `1` = Spam, `0` = Non-Spam / Ham)
- **Feature Breakdown**:
  1. **48 continuous attributes** of type `word_freq_WORD`: Percentage of words in the email matching a specific keyword (e.g., `make`, `address`, `all`, `3d`, `our`, `over`, `remove`, `internet`, `order`, `mail`, `receive`, `will`, `people`, `report`, `addresses`, `free`, `business`, `email`, `you`, `credit`, `your`, `font`, `000`, `money`, `hp`, `hpl`, `george`, `650`, `lab`, `labs`, `telnet`, `857`, `data`, `415`, `85`, `technology`, `1999`, `parts`, `pm`, `direct`, `cs`, `meeting`, `original`, `project`, `re`, `edu`, `table`, `conference`).
  2. **6 continuous attributes** of type `char_freq_CHAR`: Percentage of characters in the email matching specified punctuation marks (`;`, `(`, `[`, `!`, `$`, `#`).
  3. **3 attributes** measuring capital letter sequences:
     - `capital_run_length_average`: Average length of uninterrupted sequences of capital letters.
     - `capital_run_length_longest`: Maximum length of uninterrupted sequences of capital letters.
     - `capital_run_length_total`: Total number of capital letters in the email.
- **Data Preprocessing**: Features were scaled using `StandardScaler` fitted strictly on the 80% training split, and a stratified 20% holdout split (921 instances) was reserved for testing and Streamlit inference (`test_data.csv`).

---

## c. GitHub Repository Link
- **GitHub Repository**: `https://github.com/uty261998-collab/bits-ml-assignment-2`
- **Live Streamlit App**: `https://bits-ml-assignment-2.streamlit.app`

---

## d. Models Used & Comparison Table

All 5 classification models were trained on the same 80% training split and evaluated on the identical 20% holdout test dataset (921 samples: 558 Non-Spam, 363 Spam).

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9294** | **0.9702** | 0.9209 | 0.8981 | **0.9093** | **0.8518** |
| **Decision Tree** | 0.9034 | 0.9266 | **0.9308** | 0.8154 | 0.8693 | 0.7976 |
| **KNN** | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| **Naive Bayes** | 0.8328 | 0.9376 | 0.7146 | **0.9587** | 0.8188 | 0.6946 |
| **Support Vector Machine** | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Best overall performer.** Achieved the highest Accuracy (**92.94%**), highest AUC (**0.9702**), highest F1-score (**0.9093**), and highest Matthews Correlation Coefficient (**0.8518**). Because text frequency features in Spambase represent high-dimensional continuous log-odds signals, a regularized linear decision boundary effectively separates spam patterns with minimal overfitting. |
| **Decision Tree** | Produced the highest Precision (**93.08%**), ensuring very few legitimate emails are incorrectly marked as spam. However, it achieved lower Recall (**81.54%**) because orthogonal axis-aligned splits struggle to capture complex non-linear combinations of word frequencies across 57 dimensions without overfitting. Limiting tree depth to 6 helped regularize the model. |
| **KNN** | Demonstrated strong, balanced metrics (Accuracy: **90.77%**, AUC: **0.9506**, F1: **0.8824**) when standardized using `StandardScaler`. KNN clusters emails effectively based on Euclidean distance in word-vector space, but suffers from slower inference latency as it computes pairwise distances against all training instances. |
| **Naive Bayes (Gaussian)** | Achieved the highest Recall (**95.87%**), catching virtually all spam emails. However, it had the lowest Precision (**71.46%**) and lowest Accuracy (**83.28%**). This is directly attributable to the Naive Bayes assumption of conditional independence between features, which is clearly violated in text data where words like *"free"*, *"money"*, and *"credit"* naturally co-occur. |
| **Support Vector Machine (RBF)** | Competitive second-best performer (Accuracy: **92.73%**, AUC: **0.9667**, Precision: **0.9277**, F1: **0.9055**, MCC: **0.8471**). The Radial Basis Function (RBF) kernel mapped the 57 continuous features into a high-dimensional feature space, creating a soft margin that handles outliers and boundary samples cleanly. |

### Overall Winner for Your Dataset:
> **Winner: Logistic Regression**  
> **Justification**: Logistic Regression is the undisputed winner on this dataset. It achieved the highest **Accuracy (92.94%)**, highest **ROC-AUC (0.9702)**, highest **F1-Score (0.9093)**, and highest **MCC (0.8518)** while maintaining a strong balance between Precision (92.09%) and Recall (89.81%). In an email filtering application, balancing precision (to avoid sending legitimate work emails to the spam folder) and recall (to protect users from malicious spam) is crucial, making Logistic Regression the most reliable and computationally lightweight model for production deployment.

---

## Repository Structure
```
project-folder/
│-- app.py                     # Interactive Streamlit Web Application
│-- train.py                   # Complete Model Training Pipeline
│-- requirements.txt           # Python dependency specifications
│-- README.md                  # Comprehensive Documentation & Metric Reports
│-- test_data.csv              # Holdout Test Dataset for Streamlit Demonstration
└── model/                     # Pretrained models & preprocessing artifacts
    ├── scaler.joblib          # Fitted StandardScaler
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

