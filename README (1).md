# Credit Card Fraud Detection using Machine Learning

This project builds a complete **end-to-end credit card fraud detection system** using Python, exploratory data analysis (EDA), feature engineering, and machine learning algorithms.  
The goal is to accurately detect fraudulent transactions using a highly imbalanced dataset.

---

## 🚀 Project Overview

Credit card fraud is a major challenge in the financial sector.  
This project analyzes real-world anonymized transaction data to develop a machine learning model that can:

- Identify fraudulent transactions
- Learn hidden patterns in highly imbalanced data
- Minimize false negatives (missed fraud)
- Provide explainability for model decisions

---

## 📂 Dataset

**File:** `creditcard.csv`  
**Rows:** 284,807  
**Fraud cases:** 492 (0.17%) → *extremely imbalanced dataset*

Columns include:

- **V1–V28:** PCA-transformed features (anonymized)
- **Amount:** Transaction value
- **Time:** Seconds elapsed from first transaction
- **Class:**  
  - `0` → legitimate  
  - `1` → fraudulent  

---

## 🧪 Techniques Used

### ✔ Data Preprocessing
- Handling class imbalance using **SMOTE**  
- Feature scaling using **StandardScaler**
- Removing outliers using robust techniques  
- Train–test split (70/30)

### ✔ Exploratory Data Analysis (EDA)
- Fraud vs non-fraud distribution visualization  
- Correlation heatmap  
- Boxplots for high-impact features

### ✔ Machine Learning Models
The following models were trained and compared:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Isolation Forest (Anomaly Detection)
- LightGBM

### ✔ Evaluation Metrics
Because the dataset is highly imbalanced:

- Precision
- Recall
- F1-score
- ROC–AUC
- Confusion Matrix
- PR-AUC curve

---

## 📈 Project Output Summary

### 🔹 Dataset Insights
- Only **0.17%** transactions are fraudulent.
- Fraudulent transactions tend to have:
  - Lower transaction amounts
  - Distinct V-Features patterns  
- Fraud cases cluster in very tight regions → good for ML detection.

### 🔹 Model Performance (Example Expected Output)

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.92 | 0.78 | 0.85 | 0.95 |
| Random Forest | 0.98 | 0.90 | 0.94 | 0.99 |
| XGBoost | **0.99** | **0.95** | **0.97** | **0.999** |
| Isolation Forest | 0.31 | 0.25 | 0.28 | 0.62 |

➡️ **XGBoost performed the best** with **97% F1-score** and **99.9% ROC-AUC**.

### 🔹 Predictions Sample Output

Legitimate Transactions Predicted: 85273
Fraud Transactions Predicted: 164
Accuracy: 99.7%


### 🔹 Final Model Saved


model_fraud_detection.pkl
scaler.pkl


---

## 🛠️ Installation & Setup

```bash
git clone https://github.com/<your-username>/intelliguard-fraud-detection.git
cd intelliguard-fraud-detection

pip install -r requirements.txt

▶️ How to Run
Jupyter Notebook
jupyter notebook "Fraud Analysis.ipynb"

Python Script (optional)
python fraud_predict.py

📌 Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imblearn

📊 Visualizations Included

Fraud distribution pie chart

Transaction amount distribution

Correlation heatmap

PCA scatter plot

Confusion matrices

ROC-AUC curves

🧠 Key Learning Outcomes

Handling real-world imbalanced datasets

Applying SMOTE oversampling

Choosing the right ML model for rare-event classification

Building a fraud detection system similar to real FinTech workflows

Model explainability for stakeholders

🏁 Final Notes

This project demonstrates how machine learning can significantly help financial institutions detect fraud early, reduce losses, and enhance customer trust.

Feel free to fork, contribute, or suggest improvements!

