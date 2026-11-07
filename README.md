# Email Spam Classification and Model Evaluation

**BITS Pilani – Work Integrated Learning Programmes Division**  
**M.Tech (AIML) – Machine Learning Assignment 2**

## 1. Project Overview

The objective of this project is to develop and evaluate multiple machine learning classification algorithms to identify whether an email is Spam or Non-Spam (Ham).

Five classification algorithms are implemented, trained and evaluated using the UCI Spambase dataset. Their performance is compared using six evaluation metrics.

A Streamlit web application is developed to allow users to upload test data, select a classifier and view its evaluation results.

## 2. Problem Statement

Email spam is a common problem that affects email users by filling inboxes with unwanted and potentially harmful messages.

The goal of this project is to build a machine learning-based email spam classification system that can automatically classify emails into two categories:

- **Spam (1):** Unwanted or unsolicited emails.
- **Non-Spam / Ham (0):** Legitimate emails.

The project compares five machine learning algorithms to identify the model that provides the best classification performance.

## 3. Dataset Description

**Dataset:** UCI Spambase Dataset

**Source:** [UCI Spambase Dataset – OpenML](https://www.openml.org/search?type=data&sort=runs&id=44)

The dataset is loaded programmatically using Scikit-learn's `fetch_openml()` function.

| Dataset property | Description |
|---|---|
| Total instances | 4,601 |
| Total features | 57 |
| Target variable | class |
| Number of classes | 2 |
| Class 0 | Non-Spam |
| Class 1 | Spam |
| Problem type | Binary Classification |

The dataset contains numerical attributes representing word frequencies, character frequencies and other email-related characteristics.

## 4. Machine Learning Algorithms

The following five classification algorithms are implemented:

1. **Logistic Regression:** A linear classification algorithm used to estimate the probability of an email belonging to a particular class.
2. **Decision Tree Classifier:** A tree-based algorithm that makes predictions using feature-based decision rules.
3. **K-Nearest Neighbors (KNN):** A distance-based classification algorithm that predicts a class based on nearby training samples.
4. **Gaussian Naive Bayes:** A probabilistic classifier based on Bayes' theorem and the Gaussian distribution assumption.
5. **Support Vector Machine (SVM):** A classification algorithm that identifies an optimal decision boundary between classes.

## 5. Implementation Methodology

The project follows these steps:

1. Load the UCI Spambase dataset using OpenML.
2. Separate input features and the target variable.
3. Divide the dataset into 80% training data and 20% testing data using stratified sampling.
4. Apply StandardScaler to standardize the input features. The scaler is fitted only on the training data to prevent data leakage.
5. Train all five classification algorithms.
6. Evaluate each model using the same holdout test dataset.
7. Save trained models and preprocessing artifacts using Joblib.
8. Develop a Streamlit application to display model evaluation results.

### Train-test split

| Dataset portion | Percentage |
|---|---:|
| Training data | 80% |
| Testing data | 20% |

The test dataset contains 921 samples.

## 6. Evaluation Metrics

The following six evaluation metrics are used:

- **Accuracy:** Measures the proportion of correctly classified emails.
- **AUC:** Measures the model's ability to distinguish between spam and non-spam emails.
- **Precision:** Measures how many emails predicted as spam are actually spam.
- **Recall:** Measures how many actual spam emails are correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classification predictions.

## 7. Model Performance Comparison

The following results were obtained from the deployed Streamlit application's model comparison table.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9034 | 0.9266 | 0.9308 | 0.8154 | 0.8693 | 0.7976 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8328 | 0.9376 | 0.7146 | 0.9587 | 0.8188 | 0.6946 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

## 8. Model-wise Observations

### Logistic Regression
- Achieved the highest accuracy of 92.94%.
- Obtained the highest AUC of 0.9702.
- Achieved the highest F1 score of 0.9093.
- Obtained the highest MCC of 0.8518.
- Demonstrated the best overall performance among the five classifiers.

### Decision Tree
- Achieved the highest precision of 93.08%.
- Obtained an accuracy of 90.34%.
- Its recall was 81.54%.
- It provides interpretable decision rules but performed below Logistic Regression overall.

### K-Nearest Neighbors
- Achieved an accuracy of 90.77%.
- Obtained an AUC of 0.9506.
- Achieved an F1 score of 0.8824.
- Demonstrated reasonably balanced classification performance.

### Gaussian Naive Bayes
- Achieved the highest recall of 95.87%.
- Obtained an accuracy of 83.28%.
- Its precision was 71.46%.
- It identifies a large proportion of actual spam emails but produces more false positives.

### Support Vector Machine
- Achieved an accuracy of 92.73%.
- Obtained an AUC of 0.9667.
- Achieved an F1 score of 0.9055.
- Performed very close to Logistic Regression.

## 9. Overall Model Selection

Based on the experimental results, **Logistic Regression is selected as the overall best-performing model**.

It achieved:

- Accuracy: 92.94%
- AUC: 0.9702
- Precision: 0.9209
- Recall: 0.8981
- F1 Score: 0.9093
- MCC: 0.8518

Although Decision Tree achieved higher precision and Naive Bayes achieved higher recall, Logistic Regression provided the strongest overall balance across the evaluation metrics.

## 10. Streamlit Web Application

A Streamlit application has been developed to provide an interactive interface for model evaluation.

### Application Features

- Upload test data in CSV format.
- Use the default held-out test dataset.
- Select one of the five trained classifiers.
- Display Accuracy, AUC, Precision, Recall, F1 Score and MCC.
- Display the confusion matrix.
- Display the classification report.
- Visualize the ROC curve.
- Compare all five classifiers using a performance comparison table.
- Explore test data and dataset information.

The application uses previously trained and saved models. Uploaded test data is used for evaluation and prediction without retraining the models.

### Deployed Application

[Open Email Spam Classification – Streamlit App](https://bits-ml-assignment-2-ank.streamlit.app/)

## 11. Project Structure

```text
bits-ml-assignment-2/
│
├── train.py
├── app.py
├── requirements.txt
├── README.md
├── SUBMISSION_GUIDE.md
├── test_data.csv
│
└── model/
    ├── scaler.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics_summary.csv
```

## 12. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- OpenML
- Git and GitHub

## 13. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/vermaankesh/bits-ml-assignment-2.git
cd bits-ml-assignment-2
```

### Step 2: Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the models

```bash
python train.py
```

This downloads the Spambase dataset through OpenML, trains the five classifiers, evaluates them and saves the model artifacts and test dataset.

### Step 5: Run the Streamlit application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in the terminal.

## 14. Conclusion

This project demonstrates the application of five machine learning classification algorithms to the email spam detection problem.

All five classifiers were evaluated using the same holdout test dataset and six evaluation metrics.

Based on the experimental results, Logistic Regression achieved the best overall performance, with an accuracy of 92.94% and an AUC of 0.9702.

The Streamlit application provides an interactive way to upload test data, select classification models and compare their evaluation results.

The project demonstrates the complete machine learning workflow, including dataset loading, preprocessing, model training, evaluation, model persistence and deployment.

## 15. Repository Information

**GitHub Repository:**  
https://github.com/vermaankesh/bits-ml-assignment-2

**Deployed Streamlit Application:**  
https://bits-ml-assignment-2-ank.streamlit.app/
