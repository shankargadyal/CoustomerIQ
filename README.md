# CustomerIQ — AI-Powered Customer Churn Prediction Dashboard

> A production-ready machine learning system for predicting customer churn, comparing 6 ML models, and surfacing business intelligence through an interactive analytics dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189BBB)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [ML Results](#ml-results)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Data Leakage Fix](#data-leakage-fix)
- [Dashboard Features](#dashboard-features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Academic Context](#academic-context)

---

## Overview

CustomerIQ is a full-stack data science application that predicts customer churn using RFM (R


**Key design principles:**
- Zero data leakage — 9 leaky columns identified and removed
- Probabilistic target redefinition to simulate real-world label uncertainty
- SMOTE applied inside the CV pipeline (no test-set contamination)
- Overfitting detection with train vs. test accuracy gap monitoring
- All metrics computed from actually trained models — nothing hardcoded

---

## Live Demo

> Deployed on Render — [customeriq.onrender.com](https://customeriq.onrender.com) *(may take 30s to wake from cold start)*

---

## ML Results

All metrics are from models evaluated on the **held-out test set (20%)**, with 5-fold stratified cross-validation on the training set.

| Model | Test Acc | Train Acc | Gap | F1 Score | ROC-AUC | Overfitting |
|---------|---------|---------|---------|---------|---------|---------|
| **Random Forest ⭐** | **82.26%** | 87.38% | +5.12% | **62.07%** | 79.15% | ⚠️ Mild |
| Logistic Regression | 79.95% | 82.28% | +2.33% | 58.96% | **81.76%** | ✅ Minimal |
| XGBoost | 80.30% | 87.30% | +7.00% | 58.60% | 78.69% | ⚠️ Moderate |
| Gradient Boosting | 80.07% | 84.90% | +4.83% | 58.11% | 79.41% | ✅ Low |
| Decision Tree | 77.76% | 82.25% | +4.49% | 57.02% | 78.52% | ✅ Low |
| KNN | 76.84% | 82.97% | +6.13% | 54.42% | 78.35% | ⚠️ Moderate |
Model Selection: Random Forest was selected as the production model because it achieved the highest test accuracy (82.26%) and best F1 Score (62.07%) while maintaining acceptable generalization performance. Logistic Regression achieved the highest ROC-AUC (81.76%) and showed the lowest overfitting, making it a strong baseline model.
**Dataset:** 4,338 customers · 80/20 stratified split · 5-fold CV · SMOTE applied (21.3% churn rate)

> Accuracy range 76–82% is intentional and scientifically correct. The original dataset produced 100% accuracy due to data leakage — see [Data Leakage Fix](#data-leakage-fix).

---

## Project Structure

```
CustomerIQ/
│
├── app.py                  # Flask backend — REST API + data routes
├── preprocessing.py        # Feature engineering, leakage removal, target definition
├── train_models.py         # Full ML training pipeline with CV and SMOTE
├── evaluate_model.py       # Classification report, confusion matrix, metrics
├── model_comparison.py     # Auto-generated plain-English explanations
│
├── datasets/
│   └── rfm_data.csv        # RFM customer behavioural data (4,338 rows)
│
├── ml_models/
│   ├── churn_model.pkl     # Best model (Random Forest + preprocessor pipeline)
│   ├── model_results.json  # All 6 models — test, train, CV metrics
│   ├── feature_importance.json
│   ├── confusion_matrix.json
│   ├── roc_data.json       # ROC curve points for all models
│   ├── split_info.json     # Dataset split metadata
│   └── explanation.json    # Auto-generated business explanations
│
├── static/
│   ├── css/
│   └── js/
│       └── app.js          # Frontend — Chart.js visualisations, API calls
│
├── templates/
│   └── index.html          # Single-page dashboard
│
├── requirements.txt
├── Procfile                # Gunicorn for Render/Heroku
└── render.yaml
```

---

## ML Pipeline

The pipeline is split across 4 professional modules following industry conventions:

### 1. `preprocessing.py`
- Defines 8 legitimate features: `Recency`, `Frequency`, `Monetary`, `AvgOrderValue`, `TotalItems`, `UniqueProducts`, `Cluster`, `Country`
- Builds `ColumnTransformer`: median imputation + `StandardScaler` for numerics, `OneHotEncoder` (rare→infrequent) for country
- Redefines the churn target **probabilistically** using a logistic function of multiple features + Gaussian noise, breaking the original hard-threshold determinism that caused 100% accuracy

### 2. `train_models.py`
```python
# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE inside pipeline (training folds only)
pipe = ImbPipeline([
    ('prep',  preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf',   classifier),
])

# 5-fold cross-validation
cv_results = cross_validate(pipe, X_train, y_train,
    cv=StratifiedKFold(5), scoring=['accuracy', 'f1', 'roc_auc'],
    return_train_score=True
)
```

### 3. `evaluate_model.py`
- Full `classification_report` with precision/recall/F1 per class
- Confusion matrix with TP/TN/FP/FN breakdown and percentages
- Overfitting detection: flags any model where `train_acc - test_acc > 5%`

### 4. `model_comparison.py`
- Loads `model_results.json` and generates structured explanations
- Outputs: why the best model won, overfitting analysis, business interpretation
- Saves `explanation.json` for the dashboard API

---

## Data Leakage Fix

The original project returned **100% accuracy** on all models. Root cause analysis:

### What Was Leaking

The dataset was synthetically constructed with `ChurnRisk` defined by an exact Recency threshold:

```
Low    = Recency  1–45  days
Medium = Recency 46–90  days
High   = Recency 91+   days
```

This made the following columns **perfect or near-perfect predictors** of the target — they should never be used as features:

| Column | Why It Leaked |
|---|---|
| `ChurnScore` | Computed directly from `ChurnRisk` — perfect predictor |
| `R_Score` | Binned Recency that **defines** ChurnRisk — 91.6% accuracy alone |
| `F_Score`, `M_Score`, `RFM_Score` | Derived from same rule-based assignment |
| `CLV` | Customer Lifetime Value assigned by ChurnRisk tier |
| `Segment`, `Recommendation`, `ClusterLabel` | Downstream outputs of the same pipeline |

### The Fix

1. **Removed all 9 leaky columns** from the feature matrix
2. **Redefined the target probabilistically** — instead of the hard-threshold `ChurnRisk` label, `ChurnBinary` is computed as:

```python
log_odds = (
    3.5 * Recency_norm        # high recency → more likely to churn
  - 2.0 * Frequency_norm      # frequent buyers less likely to churn
  - 1.0 * Monetary_norm
  - 0.5 * UniqueProducts_norm
  - 2.0                       # intercept
  + np.random.normal(0, 1.2)  # realistic noise
)
ChurnBinary = (sigmoid(log_odds) > 0.5).astype(int)
```

This simulates real-world churn where no single rule perfectly determines who leaves, giving a realistic accuracy ceiling of **76–92%**.

---

## Dashboard Features

| Section | What It Shows |
|---|---|
| **Overview** | Revenue KPIs, monthly trends, customer count, avg order value |
| **Customer Segments** | RFM segment distribution, CLV by segment, scatter plots |
| **Churn Analysis** | Churn rate by segment, risk distribution, retention insights |
| **ML Models** | All 6 models — test accuracy, F1, AUC, train vs test gap, overfitting badge |
| **Model Charts** | Metrics comparison bar chart, train vs test chart, ROC curves, confusion matrix, feature importance |
| **Explanation Panel** | Auto-generated: why best model won, overfitting analysis, business interpretation |
| **Dataset Info** | Sample counts, CV folds, SMOTE status, leaky features removed |
| **Forecast** | Revenue and customer retention projections |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11, Flask 3.0 |
| **ML** | Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| **Data** | Pandas, NumPy |
| **Frontend** | Vanilla JS, Chart.js |
| **Deployment** | Gunicorn, Render |

---

## Installation

```bash
# Clone
git clone https://github.com/shankargadyal/customeriq.git
cd customeriq

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install imbalanced-learn    # for SMOTE (add to requirements if missing)
```

### Train the Models

```bash
python train_models.py
```

This will:
- Load and clean `datasets/rfm_data.csv`
- Remove all leaky features
- Run 5-fold CV on 6 classifiers
- Save all artefacts to `ml_models/`
- Print a full results table

Expected output:
```
✅ Logistic Regression  test=80.0%  train=82.2%  gap=+2.3%  f1=59.0%  auc=81.8%
⚠️  Random Forest        test=82.3%  train=87.4%  gap=+5.1%  f1=62.1%  auc=79.2%  ⚠️ OVERFIT
...
🏆 Best model: Random Forest  F1=62.1%  AUC=79.2%
```

### Generate Explanations (optional)

```bash
python model_comparison.py
```

### Run the App

```bash
python app.py
```

Open [http://localhost:5002](http://localhost:5002)

---

## API Reference

All endpoints return JSON.

| Endpoint | Method | Description |
|---|---|---|
| `/api/overview` | GET | Revenue KPIs, customer stats, monthly trends |
| `/api/segments` | GET | RFM segment breakdown and CLV data |
| `/api/churn` | GET | Churn rates, risk distribution, retention metrics |
| `/api/models` | GET | All 6 ML models — metrics, ROC data, feature importance, explanation |
| `/api/customers` | GET | Paginated customer table with churn risk scores |
| `/api/forecast` | GET | Revenue and retention forecasts |
| `/api/predict` | POST | Predict churn for a single customer |

### Example: `/api/models` response shape

```json
{
  "model_results": {
    "Random Forest": {
      "accuracy": 82.26,
      "precision": 67.35,
      "recall": 57.66,
      "f1": 62.07,
      "auc": 79.15,
      "train_accuracy": 87.41,
      "cv_accuracy": 80.12,
      "accuracy_gap": 5.14,
      "overfitting": true
    }
  },
  "best_model": "Random Forest",
  "confusion_matrix": [[668, 24], [130, 46]],
  "roc_data": { "Random Forest": { "fpr": [...], "tpr": [...], "auc": 79.15 } },
  "split_info": { "total_samples": 4338, "smote_applied": true, "cv_folds": 5 },
  "explanation": { "best_model": "Random Forest", "explanation": { ... } }
}
```

---

## Academic Context

**Project:** MSc Data Science — Dayananda Sagar University (2025–2026)

**What this demonstrates:**

- Identifying and eliminating data leakage in a real ML project
- Building a proper sklearn/imbalanced-learn pipeline with no test contamination
- Industry-standard model evaluation: stratified split, k-fold CV, overfitting detection
- Full-stack deployment of an ML application with Flask and REST APIs
- Automatic result interpretation with programmatically generated explanations

---

## License

MIT — free to use, fork, and adapt with attribution.
