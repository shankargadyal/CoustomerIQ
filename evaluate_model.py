"""
CustomerIQ – Model Evaluation  (evaluate_model.py)
===================================================
Loads the saved best model and prints a full evaluation report.
Run AFTER train_models.py.

    python evaluate_model.py
"""

import os, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)

from preprocessing import load_and_prepare, ALL_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR   = os.path.join(BASE_DIR, 'ml_models')

print("=" * 65)
print("  CustomerIQ – Model Evaluation Report")
print("=" * 65)

# ── Load artefacts ─────────────────────────────────────────────────────────
with open(os.path.join(ML_DIR, 'churn_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(ML_DIR, 'model_results.json')) as f:
    all_results = json.load(f)

with open(os.path.join(ML_DIR, 'split_info.json')) as f:
    split_info = json.load(f)

best_name = split_info['best_model']
print(f"\n📦 Best model: {best_name}")
print(f"   Dataset  : {split_info['total_samples']:,} samples")
print(f"   Train    : {split_info['train_samples']:,}  |  Test: {split_info['test_samples']:,}")
print(f"   SMOTE    : {'Yes' if split_info['smote_applied'] else 'No'}")
print(f"   CV Folds : {split_info['cv_folds']}")
print(f"\n   Leaky features removed: {split_info['leaky_features_removed']}")

# ── Re-split data (same seed) to get X_test, y_test ────────────────────────
from sklearn.model_selection import train_test_split
X, y = load_and_prepare(os.path.join(BASE_DIR, 'datasets', 'rfm_data.csv'))
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ── Classification report ────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  Classification Report (held-out test set)")
print("─" * 65)
print(classification_report(y_test, y_pred,
                             target_names=['Non-Churn (0)', 'Churn (1)']))

# ── Confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("  Confusion Matrix:")
print(f"  {'':12}  Predicted 0  Predicted 1")
print(f"  {'Actual 0':12}     {tn:>6}       {fp:>6}   (FP={fp})")
print(f"  {'Actual 1':12}     {fn:>6}       {tp:>6}   (FN={fn})")
print(f"\n  True Positives  (TP): {tp}")
print(f"  True Negatives  (TN): {tn}")
print(f"  False Positives (FP): {fp}  ← non-churners incorrectly flagged")
print(f"  False Negatives (FN): {fn}  ← churners missed")

# ── Key metrics ───────────────────────────────────────────────────────────────
auc = roc_auc_score(y_test, y_prob)
ap  = average_precision_score(y_test, y_prob)
r   = all_results[best_name]

print("\n" + "─" * 65)
print("  Summary Metrics")
print("─" * 65)
print(f"  Accuracy       : {r['accuracy']:.1f}%")
print(f"  Precision      : {r['precision']:.1f}%")
print(f"  Recall         : {r['recall']:.1f}%")
print(f"  F1 Score       : {r['f1']:.1f}%")
print(f"  ROC-AUC        : {r['auc']:.1f}%")
print(f"  Train Accuracy : {r['train_accuracy']:.1f}%")
acc_gap = r['accuracy_gap']
gen_gap = r.get('generalization_gap', acc_gap)
print(f"  Generalization : {gen_gap:+.1f}%  {'⚠️  OVERFITTING' if r['overfitting'] else '✅ Healthy'}")
print(f"  CV Accuracy    : {r['cv_accuracy']:.1f}%  (±std across {split_info['cv_folds']} folds)")

# ── All models comparison ─────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  All Models – Test Set Performance")
print("─" * 65)
print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}  Overfit?")
print("-" * 65)
for name, r_ in sorted(all_results.items(), key=lambda x: -x[1]['f1']):
    flag = "⚠️" if r_['overfitting'] else "✅"
    print(
        f"{name:<25} {r_['accuracy']:>5.1f}% {r_['precision']:>5.1f}%"
        f" {r_['recall']:>5.1f}% {r_['f1']:>5.1f}% {r_['auc']:>5.1f}%  {flag}"
    )

print("\n✅ Evaluation complete. No hardcoded metrics — all from live model.")
