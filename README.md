# Credit Card Fraud Detection Capstone

This repository contains a Jupyter Notebook version of a credit card fraud detection capstone project. The goal is to identify fraudulent transactions from highly imbalanced transaction data and compare multiple machine learning approaches.

## Project Overview

The notebook covers the full workflow:
- data loading for train and test datasets
- feature engineering from transaction timestamp and customer date of birth
- distance calculation between customer and merchant locations
- categorical encoding for model-ready data
- class imbalance handling using SMOTE and ADASYN
- model training with Logistic Regression, Decision Tree, and Random Forest
- evaluation using Recall, ROC-AUC, confusion matrix, and classification report
- simple business cost analysis before and after model deployment

## Dataset
This project uses the standard fraud dataset (Source: Kaggle ) with these columns:
- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

Target column:
- `Class = 0` → non-fraud transaction
- `Class = 1` → fraud transaction

## What the notebook does
1. Loads `creditcard.csv`
2. Performs quick checks on schema and class balance
3. Splits the data into **train** and **test** sets using **stratified sampling**
4. Applies preprocessing with `StandardScaler`
5. Trains baseline models:
   - Logistic Regression
   - Random Forest
6. Evaluates model performance using:
   - Classification report
   - Confusion matrix
   - ROC-AUC
   - PR-AUC

## Train-test split used
The notebook uses:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

This keeps the fraud ratio similar in both train and test datasets.


## Files

- `creditcard_fraud_detection.ipynb` - notebook version of the code
- `README.md` - project documentation for GitHub



## How to Run in Jupyter Notebook

1. Clone or download the repository.
2. Create a virtual environment if needed.
3. Install the required Python libraries.
4. Add the dataset files inside the `data/` folder.
5. Launch Jupyter Notebook or JupyterLab.
6. Open `creditcard_fraud_detection.ipynb` and run the cells in order.

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn statsmodels jupyter
```

### Start Jupyter

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

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
