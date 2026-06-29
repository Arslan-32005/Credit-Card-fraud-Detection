# Credit Card Fraud Detection

# Overview
Binary classification project to detect fraudulent credit card transactions using machine learning. The dataset is highly imbalanced (99.8% normal, 0.2% fraud), making this a real-world class imbalance problem.

# Dataset
Source: Kaggle — ULB Machine Learning Group
Size: 284,807 transactions, 31 features
Class Distribution: 284,315 normal vs 492 fraud (0.17% fraud)
Features: V1-V28 (PCA transformed for privacy), Time, Amount

# Problem
Standard accuracy is misleading on imbalanced data — a model predicting "always normal" achieves 99.8% accuracy but detects zero fraud. Focus was on **Recall** and **F1-score** for fraud class.

# Approach

# 1. Preprocessing
Dropped `Class` column as target variable
Applied `StandardScaler` on `Time` and `Amount` — V1-V28 already PCA-scaled
Train-test split: 80/20 with `stratify=y` to maintain class ratio

# 2. Handling Class Imbalance — SMOTE
Applied SMOTE (Synthetic Minority Oversampling Technique) on training data only
Fraud samples increased from 394 → 227,451
Test data kept original and imbalanced — to reflect real-world performance

# 3. Models Compared
| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) |
|-------|------------------|----------------|------------|
| Logistic Regression | 6% | 92% | 0.11 |
| Random Forest | 85% | 84% | 0.84 |
| XGBoost | 68% | 86% | 0.76 |

# 4. Final Model — Random Forest
Random Forest outperformed both LR and XGBoost on test set
- **F1: 0.84, Precision: 85%, Recall: 84%**
- GridSearchCV was intentionally skipped for RF — computational cost on 227K SMOTE-balanced rows was 2+ hours. RF with default params already achieved best test performance, confirming tuning was not necessary for this dataset.

# Key Learnings
- SMOTE must only be applied on training data — applying on test data would create synthetic fraud cases that don't exist in real world, giving fake performance metrics
- Accuracy is a misleading metric for imbalanced datasets — F1 and Recall are better indicators
- Logistic Regression had high Recall (92%) but terrible Precision (6%) — model was flagging almost every transaction as fraud
- Random Forest generalized better on real imbalanced test data compared to XGBoost

# Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler, SMOTE)
- XGBoost
- imbalanced-learn

## Files
- `Credit_Card_Fraud.ipynb` — complete notebook
- `creditcard.csv` — dataset (not pushed, download from Kaggle)
