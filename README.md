# Fraud and Credit Risk Detection using Machine Learning

This project applies machine learning techniques to detect fraudulent transactions and analyze credit risk using structured financial data. The focus is on understanding imbalanced classification problems and evaluating model performance using appropriate metrics.

---

## Project Overview

Financial fraud detection is challenging due to the low occurrence of fraudulent transactions compared to legitimate ones.  
This project explores how different machine learning models behave on imbalanced financial data and how evaluation metrics reflect risk-related trade-offs.

---

## Dataset

- Structured financial transaction dataset (~135,000 records)
- Binary target variable indicating fraudulent vs non-fraudulent transactions
- Dataset exhibits strong class imbalance

---

## Data Preprocessing and Analysis

- Loaded and inspected structured transaction data
- Handled missing values and validated data types
- Performed feature scaling and train–test split
- Analyzed class imbalance and its impact on model behavior

---

## Exploratory Data Analysis (EDA)

- Examined distribution of fraud vs non-fraud transactions
- Analyzed feature relationships and correlations
- Used visualizations to understand patterns relevant to fraud detection

---

## Machine Learning Models

The following classification models were implemented and evaluated:

- Logistic Regression  
- K-Nearest Neighbors (KNN)

---

## Model Evaluation

Due to the imbalanced nature of the dataset, model performance was evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

Model results were compared to understand differences in fraud detection sensitivity and misclassification behavior.

---

## Key Insights

- Model performance varies significantly under class imbalance.
- KNN achieved better overall test performance compared to logistic regression.
- Fraud-class recall highlighted the importance of minimizing false negatives in financial risk scenarios.
- Confusion matrix analysis helped interpret trade-offs between fraud detection sensitivity and false positives.

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

## Notes

This project was completed as part of academic work and demonstrates applied machine learning techniques for fraud detection and credit risk analysis.
