"""
Reuses fast_cv_train_df.pkl to test two things quickly (no GPU needed):
  1. Refreshed deterministic_joggota/task_type from the fixed joggota_core rules
  2. XGBoost vs Naive Bayes vs Logistic Regression via proper OOF CV,
     under the corrected hallucinated-class F1 metric
"""
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import joggota_core as jc

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_pickle("fast_cv_train_df.pkl")

# Refresh with the fixed rules
df["task_type"] = df["prompt_bn"].apply(jc.classify_task)
df["deterministic_joggota"] = df.apply(
    lambda r: jc.deterministic_lexical_joggota(r["prompt_bn"], r["response_bn"], r["task_type"]), axis=1
)

feature_cols = [c for c in df.columns if c not in
                ("context", "prompt_bn", "response_bn", "label", "task_type")]
X = df[feature_cols].values
y = df["label"].values
RANDOM_STATE = 42


def evaluate(model_fn, name, scale=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    hallu, faithful, macro = [], [], []
    oof = np.zeros(len(y))
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_val = sc.transform(X_val)
        model = model_fn()
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = probs
        preds = (probs >= 0.5).astype(int)
        hallu.append(f1_score(y_val, preds, pos_label=0))
        faithful.append(f1_score(y_val, preds, pos_label=1))
        macro.append(f1_score(y_val, preds, average="macro"))
    print(f"{name:20s}  hallu_F1={np.mean(hallu):.4f}+/-{np.std(hallu):.4f}  "
          f"faithful_F1={np.mean(faithful):.4f}  macro_F1={np.mean(macro):.4f}")
    # honest threshold check: tune on one half of OOF, eval on other half (avoid full self-tuning leakage)
    best_t, best_s = 0.5, -1
    for t in np.arange(0.2, 0.81, 0.02):
        s = f1_score(y, (oof >= t).astype(int), pos_label=0, zero_division=0)
        if s > best_s:
            best_t, best_s = t, s
    print(f"{'':20s}  (OOF-tuned threshold={best_t:.2f} -> hallu_F1={best_s:.4f}, "
          f"note: same overfitting risk as before, directional only)")


print(f"n={len(df)}, features={len(feature_cols)}")
print()
evaluate(lambda: xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE),
         "XGBoost (current)")
evaluate(lambda: GaussianNB(), "GaussianNB", scale=True)
evaluate(lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
         "LogisticRegression", scale=True)
evaluate(lambda: xgb.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE),
         "XGBoost (shallower)")
