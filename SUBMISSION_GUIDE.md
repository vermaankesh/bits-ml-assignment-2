# BITS Pilani M.Tech (AIML) — Assignment 2 Submission Guide

This guide walks you through the exact steps to push your repository, deploy the Streamlit web application, run in BITS Virtual Lab for your screenshot, and submit the final PDF.

---

## Step 1: Push Code to Your Personal GitHub Account

Open PowerShell or Terminal and navigate to the project directory:

```bash
cd E:\UtsavPath\v6_18_july\BITS_ML_Assignment_2
```

Initialize Git and push to your GitHub profile:
```bash
git init
git add .
git commit -m "Complete ML Assignment 2: Models, Streamlit App & Documentation"
git branch -M main
```

Now, go to [GitHub.com](https://github.com), click **New Repository**, name it (e.g., `bits-ml-assignment-2`), keep it **Public**, and run:

```bash
git remote add origin https://github.com/uty261998-collab/bits-ml-assignment-2.git
git push -u origin main
```

*(Verify on GitHub that `app.py`, `train.py`, `requirements.txt`, `README.md`, `test_data.csv`, and `model/` are all visible).*

---

## Step 2: Deploy to Streamlit Community Cloud (FREE)

1. Go to **[https://streamlit.io/cloud](https://streamlit.io/cloud)** and click **Sign in** (choose **Continue with GitHub**).
2. Click the **"New App"** or **"Create App"** button at the top right.
3. In the deployment dialog:
   - **Repository**: Select your repository `your-username/bits-ml-assignment-2`.
   - **Branch**: `main`.
   - **Main file path**: `app.py`.
   - **App URL**: Choose a custom name if available (e.g., `bits-ml-spam-classifier.streamlit.app`).
4. Click **Deploy!**
5. Wait 1–2 minutes for Streamlit to install `requirements.txt` and boot the app.
6. Verify your live app:
   - The default `test_data.csv` loads automatically.
   - You can upload `test_data.csv` via the file uploader.
   - Selecting each of the 5 models updates the metrics, confusion matrix, classification report, and comparison table.
7. **Copy your live app URL** (e.g. `https://bits-ml-spam-classifier.streamlit.app`).

---

## Step 3: Run on BITS Virtual Lab & Take Screenshot (Mandatory 1 Mark)

1. Log into **BITS Virtual Lab** at your assigned portal (`csislab`).
2. Inside the BITS Lab environment (Jupyter / Terminal):
   - Clone your repository:
     ```bash
     git clone https://github.com/<YOUR_GITHUB_USERNAME>/bits-ml-assignment-2.git
     cd bits-ml-assignment-2
     ```
   - Run the training script:
     ```bash
     python train.py
     ```
3. **Capture a full-screen screenshot**:
   - Ensure your BITS Virtual Lab browser tab / user identifier is visible.
   - Ensure the terminal output displaying the 5-model evaluation table is clearly visible.
   - Save the image as `bits_lab_screenshot.png`.

---

## Step 4: Assemble the Single PDF Submission

The rubric strictly requires a **single PDF file** with contents in this exact order:

### 1. GitHub Repository Link (Clickable)
`https://github.com/<YOUR_GITHUB_USERNAME>/bits-ml-assignment-2`

### 2. Live Streamlit App Link (Clickable)
`https://<YOUR_APP_NAME>.streamlit.app`

### 3. Screenshot Proof
Attach the full screenshot of your execution inside BITS Virtual Lab.

### 4. Complete README.md Content
Copy and paste the exact text from `README.md` (sections **a**, **b**, **c**, and **d** including the Comparison Table and Observations Table).

---

## Final Submission Checklist

- [x] Dataset has 57 features (≥12) and 4,601 instances (≥500).
- [x] All 5 classifiers implemented (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM).
- [x] All 6 metrics computed (Accuracy, AUC, Precision, Recall, F1, MCC).
- [x] Pre-trained models saved in `model/` (no retraining on user upload).
- [x] Confusion matrix & Classification report visible in Streamlit.
- [x] BITS Virtual Lab execution screenshot included in PDF.
- [x] GitHub repo is public and contains all files.
- [x] Submit before **08-Nov-2026 23:59 PM**.

