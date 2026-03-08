# Credit Card Fraud Detection Capstone

This repository contains a Jupyter Notebook version of machine learning project for **fraud detection on highly imbalanced transaction data** using `creditcard.csv`.

It focuses on:
- practical fraud detection modeling
- train/test splitting from a single dataset
- imbalance handling with **SMOTE** and **ADASYN**
- model comparison across linear and tree-based methods
- business-facing **cost-benefit analysis**

---

## Project overview

This project uses the public **credit card fraud detection** dataset with:
- **284,807** transactions
- **31** columns total
- **492** fraud cases
- fraud rate of about **0.173%**

The dataset is highly imbalanced, which makes this a strong classification case study for discussing:
- why accuracy can be misleading
- how to evaluate rare-event models
- how business costs affect threshold choice

---

## Problem statement

The goal is to predict whether a transaction is fraudulent.

This is not just a standard classification problem. It is an **imbalanced fraud detection** problem where:
- fraud cases are very rare
- missing a fraud can be expensive
- flagging too many good transactions creates friction and review costs

Because of that, the project emphasizes:
- **Recall**
- **F1-score**
- **ROC-AUC**
- **PR-AUC**
- business cost trade-offs between false positives and false negatives

---

## Dataset details

The dataset contains:
- `Time`
- `V1` to `V28`
- `Amount`
- `Class` (target)

### Target meaning
- `0` = non-fraud
- `1` = fraud

### Notes
- `V1` to `V28` are anonymized transformed features
- `Time` is the number of seconds elapsed
- `Amount` is the transaction amount
- since the dataset is already mostly numeric, the notebook keeps preprocessing practical and lightweight

---

## Feature engineering used

Even though the dataset is anonymized, the notebook adds a few business-style derived variables to make the workflow more explainable:

- `Amount_log`  
  Reduces skew in transaction amounts

- `Time_days`  
  Converts time from seconds to days

- `Hour`  
  Approximate hour of transaction

- `Hour_sin` and `Hour_cos`  
  Cyclical time-of-day encoding

- `Amount_zscore`  
  Standardized amount signal

- `Amount_is_zero`  
  Flags zero-amount edge cases

These features help show interviewers that the project was not limited to “just run a model.”

---

## Modeling approach

The notebook compares multiple models:

### 1. Baseline Logistic Regression
Used as a simple, interpretable starting point.

### 2. Logistic Regression with SMOTE
Adds synthetic fraud examples to improve minority class learning.

### 3. Logistic Regression with ADASYN
Focuses more heavily on harder-to-learn minority samples.

### 4. Decision Tree models
Captures non-linear relationships and is easy to explain.

### 5. Random Forest models
Provides a stronger tabular baseline with better robustness and interaction handling.

---

## Why these evaluation metrics matter

For fraud detection, **accuracy alone is not enough**.

A model can achieve very high accuracy simply by predicting “not fraud” for almost every transaction, because fraud cases are so rare.

That is why this project focuses on:
- **Recall** → how many fraud cases were found
- **Precision** → how many flagged transactions were actually fraud
- **F1-score** → balance between precision and recall
- **ROC-AUC** → ranking quality across thresholds
- **PR-AUC** → especially useful for rare-event detection

---

## Train/test split

The dataset starts as a **single CSV file**, so the notebook creates its own split:

```python
train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
```

### Why stratify?
Because fraud is rare, stratification preserves a similar fraud ratio in both train and test sets.

---

## How to run

### 1. Create a project folder
Example structure:

```text
project/
├── creditcard_fraud_detection.ipynb
├── creditcard.csv
├── README.md
└── COST_BENEFIT_ANALYSIS.md
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

### 4. Open the notebook
Run:

`creditcard_fraud_detection.ipynb`

---

## Notebook Structure

1. Imports and display settings  
2. File paths  
3. Load datasets  
4. Helper functions  
5. Feature engineering  
6. Categorical encoding  
7. Prepare modeling dataset  
8. Exploratory analysis  
9. Feature selection  
10. Scaling for Logistic Regression  
11. Baseline Logistic Regression  
12. Logistic Regression with SMOTE and ADASYN  
13. Decision Tree models  
14. Random Forest models  
15. Model comparison  
16. Holdout test evaluation  
17. Fraud prediction sample output  
18. Business cost analysis  
19. Final notes and next steps  



## Recommended Repository Structure

```text
project-root/
├── README.md
├── creditcard_fraud_detection.ipynb
└── creditcard.csv
```
