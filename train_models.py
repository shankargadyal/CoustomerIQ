"""
CustomerIQ – Model Training  (train_models.py)
===============================================
Trains six classifiers on leakage-free features with:
  • Stratified 80/20 train/test split
  • 5-fold stratified cross-validation
  • SMOTE oversampling on training fold only (inside CV loop via Pipeline)
  • Training vs test accuracy comparison (overfitting detection)
  • Realistic accuracy range expected: 70 – 92 %

Run:
    python train_models.py
"""

import os, json, pickle, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

from preprocessing import load_and_prepare, build_preprocessor, get_feature_names, ALL_FEATURES

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
ML_DIR       = os.path.join(BASE_DIR, 'ml_models')
os.makedirs(ML_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS     = 5
GENERALIZATION_GAP_THRESHOLD = 8.0

print("=" * 65)
print("  CustomerIQ – ML Training Pipeline (leakage-free)")
print("=" * 65)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\n📂 Loading & preparing data...")
csv_path = os.path.join(DATASETS_DIR, 'rfm_data.csv')
X, y = load_and_prepare(csv_path)
print(f"   Samples  : {len(X):,}")
print(f"   Features : {X.shape[1]}")
print(f"   Churn=1  : {y.sum():,} ({y.mean()*100:.1f}%)  Non-churn=0: {(~y.astype(bool)).sum():,}")

# ── 2. Train / test split ─────────────────────────────────────────────────────
print("\n✂️  Splitting data (stratified 80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"   Train : {len(X_train):,}  |  Test : {len(X_test):,}")
print(f"   Train churn rate: {y_train.mean()*100:.1f}%  |  Test: {y_test.mean()*100:.1f}%")

# ── 3. SMOTE decision ─────────────────────────────────────────────────────────
class_ratio = y_train.mean()
use_smote = class_ratio < 0.4 or class_ratio > 0.6
print(f"\n⚖️  Churn rate = {class_ratio*100:.1f}% → SMOTE {'enabled' if use_smote else 'not needed'}")

# ── 4. Model definitions ──────────────────────────────────────────────────────
models_def = {
    'Logistic Regression': LogisticRegression(max_iter=1200, random_state=RANDOM_STATE, C=0.8, class_weight='balanced'),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, min_samples_split=40, random_state=RANDOM_STATE),
    'Random Forest':       RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=10, min_samples_split=20, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost':             xgb.XGBClassifier(n_estimators=140, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.5, reg_lambda=1.5, gamma=0.1, random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
    'KNN':                 KNeighborsClassifier(n_neighbors=25, weights='distance'),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=120, max_depth=2, min_samples_leaf=15, learning_rate=0.04, subsample=0.85, random_state=RANDOM_STATE),
}

# ── 5. Training loop ──────────────────────────────────────────────────────────
print(f"\n🤖 Training {len(models_def)} models with {CV_FOLDS}-fold CV...\n")

results = {}
trained = {}
cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for name, clf in models_def.items():
    t0 = time.time()
    preprocessor = build_preprocessor()

    if use_smote:
        pipe = ImbPipeline([('prep', preprocessor), ('smote', SMOTE(random_state=RANDOM_STATE)), ('clf', clf)])
    else:
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])

    cv_res = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring=['accuracy', 'f1', 'roc_auc'],
                            return_train_score=True, n_jobs=-1)

    cv_test_acc  = cv_res['test_accuracy'].mean()
    cv_train_acc = cv_res['train_accuracy'].mean()
    cv_f1        = cv_res['test_f1'].mean()
    cv_auc       = cv_res['test_roc_auc'].mean()
    cv_gap       = cv_train_acc - cv_test_acc

    pipe.fit(X_train, y_train)
    y_pred     = pipe.predict(X_test)
    y_prob     = pipe.predict_proba(X_test)[:, 1]
    y_pred_tr  = pipe.predict(X_train)

    test_acc  = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_pred_tr)
    prec      = precision_score(y_test, y_pred, zero_division=0)
    rec       = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_prob)
    acc_gap   = (train_acc - test_acc) * 100
    generalization_gap = max(acc_gap, cv_gap * 100)
    overfit   = generalization_gap > GENERALIZATION_GAP_THRESHOLD
    elapsed   = round(time.time() - t0, 3)

    results[name] = {
        'accuracy':          round(test_acc  * 100, 2),
        'precision':         round(prec      * 100, 2),
        'recall':            round(rec       * 100, 2),
        'f1':                round(f1        * 100, 2),
        'auc':               round(auc       * 100, 2),
        'train_accuracy':    round(train_acc * 100, 2),
        'cv_accuracy':       round(cv_test_acc  * 100, 2),
        'cv_f1':             round(cv_f1        * 100, 2),
        'cv_auc':            round(cv_auc       * 100, 2),
        'cv_train_accuracy': round(cv_train_acc * 100, 2),
        'cv_accuracy_gap':   round(cv_gap * 100, 2),
        'accuracy_gap':      round(acc_gap, 2),
        'generalization_gap': round(generalization_gap, 2),
        'overfitting':       bool(overfit),
        'train_time':        elapsed,
    }
    trained[name] = pipe

    flag = " ⚠️  OVERFIT" if overfit else ""
    print(f"   {'⚠️ ' if overfit else '✅'} {name:<22}  test={test_acc*100:.1f}%  train={train_acc*100:.1f}%  gap={acc_gap:+.1f}%  f1={f1*100:.1f}%  auc={auc*100:.1f}%{flag}")

# ── 6. Best model ─────────────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['f1'])
best_pipe = trained[best_name]
print(f"\n🏆 Best model: {best_name}  F1={results[best_name]['f1']}%  AUC={results[best_name]['auc']}%")

# ── 7. Confusion matrix ───────────────────────────────────────────────────────
y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best).tolist()

# ── 8. Feature importance ─────────────────────────────────────────────────────
fi_json = []
for tm in ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Decision Tree']:
    if tm not in trained:
        continue
    pipe_tm = trained[tm]
    clf_step = pipe_tm.named_steps.get('clf') or pipe_tm[-1]
    if hasattr(clf_step, 'feature_importances_'):
        prep_fitted = pipe_tm.named_steps.get('prep') or pipe_tm[0]
        feat_names  = get_feature_names(prep_fitted)
        fi_json = sorted(zip(feat_names, clf_step.feature_importances_.tolist()),
                         key=lambda x: x[1], reverse=True)
        print(f"\n📌 Feature importances ({tm}):")
        for f_, imp in fi_json[:8]:
            print(f"   {f_:<25} {imp:.4f}  {'█'*int(imp*40)}")
        break

# ── 9. ROC curves ─────────────────────────────────────────────────────────────
roc_data = {}
for name, pipe in trained.items():
    yp = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, yp)
    # downsample to ~100 points for compact JSON
    step = max(1, len(fpr) // 100)
    roc_data[name] = {
        'fpr': [round(v, 4) for v in fpr[::step].tolist()],
        'tpr': [round(v, 4) for v in tpr[::step].tolist()],
        'auc': results[name]['auc'],
    }

# ── 10. Save artefacts ────────────────────────────────────────────────────────
print("\n💾 Saving artefacts...")

with open(os.path.join(ML_DIR, 'churn_model.pkl'), 'wb') as f:
    pickle.dump(best_pipe, f)
with open(os.path.join(ML_DIR, 'features.json'), 'w') as f:
    json.dump(ALL_FEATURES, f)
with open(os.path.join(ML_DIR, 'model_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
with open(os.path.join(ML_DIR, 'feature_importance.json'), 'w') as f:
    json.dump(fi_json, f, indent=2)
with open(os.path.join(ML_DIR, 'confusion_matrix.json'), 'w') as f:
    json.dump(cm, f)
with open(os.path.join(ML_DIR, 'roc_data.json'), 'w') as f:
    json.dump(roc_data, f)

split_info = {
    'total_samples': int(len(X)),
    'train_samples': int(len(X_train)),
    'test_samples':  int(len(X_test)),
    'churn_rate_train': round(float(y_train.mean()) * 100, 1),
    'churn_rate_test':  round(float(y_test.mean()) * 100, 1),
    'features_used': ALL_FEATURES,
    'leaky_features_removed': ['ChurnScore','R_Score','F_Score','M_Score','RFM_Score','CLV','Segment','Recommendation','ClusterLabel'],
    'smote_applied': bool(use_smote),
    'cv_folds': CV_FOLDS,
    'best_model': best_name,
}
with open(os.path.join(ML_DIR, 'split_info.json'), 'w') as f:
    json.dump(split_info, f, indent=2)

# dummy scaler.pkl for backward compat
from sklearn.preprocessing import StandardScaler as _SS
_sc = _SS(); _sc.fit([[0],[1]])
with open(os.path.join(ML_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(_sc, f)

print("   ✅ churn_model.pkl, model_results.json, feature_importance.json")
print("   ✅ confusion_matrix.json, roc_data.json, split_info.json")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"{'Model':<25} {'Test':>7} {'Train':>7} {'Gap':>6} {'F1':>6} {'AUC':>6}  Overfit?")
print("-"*70)
for name, r in sorted(results.items(), key=lambda x: -x[1]['f1']):
    flag = "⚠️ YES" if r['overfitting'] else "✅ NO "
    print(f"{name:<25} {r['accuracy']:>6.1f}% {r['train_accuracy']:>6.1f}% {r['accuracy_gap']:>+5.1f}% {r['f1']:>5.1f}% {r['auc']:>5.1f}%  {flag}")
print(f"\n🏆 Best: {best_name}  |  📁 {ML_DIR}")
print(f"{'='*65}")
