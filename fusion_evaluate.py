"""
Loads fusion_train_features.pkl / fusion_test_features.pkl (produced by fusion_pipeline.py)
and does the honest part: proper OOF cross-validation, no in-sample leakage, no
double-dipped threshold search -- comparing his features / ours / combined across
three model types, before picking a final submission.

Threshold handling (learned the hard way tonight): a single threshold grid-searched
against the full OOF pool overfits on 299 rows. Instead this averages each fold's own
best threshold (found only on that fold's held-out slice) -- still not perfect with this
little data, but meaningfully less prone to cherry-picking noise than a global search.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

train = pd.read_pickle("fusion_train_features.pkl")
test = pd.read_pickle("fusion_test_features.pkl")

HIS_FEATURES = ["llm", "pA", "pB", "judge_disagreement", "nli_summac_strict",
                "nli_summac_soft", "math_override", "has_context"]
OUR_FEATURES = ["context_containment", "novel_char_ratio", "word_entropy", "char_entropy",
                "token_overlap_ctx_resp", "length_ratio", "deterministic_joggota",
                "cultural_default_flag", "resp_is_refusal", "resp_code_switch_ratio",
                "resp_repetition_score", "resp_is_question",
                "nli_summac_soft_en", "cross_lingual_disagreement"]
ALL_FEATURES = HIS_FEATURES + OUR_FEATURES

y = train["label"].values
RANDOM_STATE = 42
N_FOLDS = 5


def get_model(kind):
    if kind == "gnb":
        return GaussianNB(), True
    if kind == "logreg":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True
    if kind == "xgb":
        return xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE), False
    raise ValueError(kind)


def evaluate(feature_cols, model_kind, name):
    X = train[feature_cols].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    fold_thresholds = []
    hallu_scores = []
    for tr_idx, val_idx in skf.split(X, y):
        model, scale = get_model(model_kind)
        X_tr, X_val = X[tr_idx], X[val_idx]
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_val = sc.transform(X_val)
        model.fit(X_tr, y[tr_idx])
        probs = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = probs
        # per-fold threshold: searched only on this fold's own held-out data
        best_t, best_s = 0.5, -1
        for t in np.arange(0.2, 0.81, 0.02):
            s = f1_score(y[val_idx], (probs >= t).astype(int), pos_label=0, zero_division=0)
            if s > best_s:
                best_t, best_s = t, s
        fold_thresholds.append(best_t)
        hallu_scores.append(f1_score(y[val_idx], (probs >= 0.5).astype(int), pos_label=0, zero_division=0))

    avg_threshold = float(np.mean(fold_thresholds))
    hallu_at_avg_t = f1_score(y, (oof >= avg_threshold).astype(int), pos_label=0, zero_division=0)
    macro_at_avg_t = f1_score(y, (oof >= avg_threshold).astype(int), average="macro", zero_division=0)
    print(f"{name:35s} n_feat={len(feature_cols):2d}  hallu_F1@0.5={np.mean(hallu_scores):.4f}  "
          f"avg_fold_thr={avg_threshold:.2f}  hallu_F1@avg_thr={hallu_at_avg_t:.4f}  macro_F1={macro_at_avg_t:.4f}")
    return oof, avg_threshold, hallu_at_avg_t


print(f"n={len(train)}, his_features={len(HIS_FEATURES)}, our_features={len(OUR_FEATURES)}, combined={len(ALL_FEATURES)}")
print()
results = {}
for feat_name, feat_cols in [("his features alone", HIS_FEATURES),
                              ("our features alone", OUR_FEATURES),
                              ("combined (fusion)", ALL_FEATURES)]:
    for model_kind in ["gnb", "logreg", "xgb"]:
        key = f"{feat_name} + {model_kind}"
        results[key] = evaluate(feat_cols, model_kind, key)

best_key = max(results, key=lambda k: results[k][2])
print(f"\nBest by OOF hallu_F1: {best_key}  ({results[best_key][2]:.4f})")

# ---- train the winning config on full data, predict test, write submission ----
feat_name, model_kind = best_key.split(" + ")
feat_cols = {"his features alone": HIS_FEATURES, "our features alone": OUR_FEATURES,
             "combined (fusion)": ALL_FEATURES}[feat_name]
_, threshold, _ = results[best_key]

X_full = train[feat_cols].values
X_test = test[feat_cols].values
model, scale = get_model(model_kind)
if scale:
    sc = StandardScaler()
    X_full = sc.fit_transform(X_full)
    X_test = sc.transform(X_test)
model.fit(X_full, y)
test_probs = model.predict_proba(X_test)[:, 1]
preds = (test_probs >= threshold).astype(int)

submission = pd.DataFrame({"id": test["id"].values, "label": preds})
submission.to_csv("submission_fusion.csv", index=False)
print(f"\nWrote submission_fusion.csv | threshold={threshold:.2f} | "
      f"balance (1=faithful): {submission['label'].value_counts(normalize=True).round(3).to_dict()}")

# also always write a pure-his-features submission as a safe fallback / comparison point
his_oof, his_thr, his_f1 = results["his features alone + gnb"]
model_h, scale_h = get_model("gnb")
X_full_h = train[HIS_FEATURES].values
X_test_h = test[HIS_FEATURES].values
if scale_h:
    sc = StandardScaler()
    X_full_h = sc.fit_transform(X_full_h)
    X_test_h = sc.transform(X_test_h)
model_h.fit(X_full_h, y)
preds_h = (model_h.predict_proba(X_test_h)[:, 1] >= his_thr).astype(int)
pd.DataFrame({"id": test["id"].values, "label": preds_h}).to_csv("submission_his_gnb_baseline.csv", index=False)
print("Also wrote submission_his_gnb_baseline.csv as a fallback comparison point.")
