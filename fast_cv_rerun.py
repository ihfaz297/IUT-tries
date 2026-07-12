"""
Reuses fast_cv_train_df.pkl (already has translation/NLI/embedding features
computed — the slow part) to:
  1. Re-run SmallLLMJudge now that `accelerate` is installed locally
  2. Recompute CV under the corrected hallucinated-class F1 metric
"""
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from submission_pipeline import SmallLLMJudge, LLM_CHECKPOINTS, RANDOM_STATE
from fast_cv import cross_validate

sys.stdout.reconfigure(encoding="utf-8")

train_df = pd.read_pickle("fast_cv_train_df.pkl")
print(f"Loaded {len(train_df)} rows with {train_df.shape[1]} columns")

print("\nRe-running SmallLLMJudge (Qwen) now that accelerate is installed...")
import os
checkpoint = next((c for c in LLM_CHECKPOINTS if os.path.exists(c)), "Qwen/Qwen2.5-1.5B-Instruct")
try:
    judge = SmallLLMJudge(checkpoint_path=checkpoint, device="cpu")
    llm_scores = judge.score_no_context_rows(train_df)
    train_df["llm_judge_score"] = np.nan_to_num(llm_scores, nan=0.0)
    llm_ok = True
    nz = train_df["llm_judge_score"][train_df["has_context"] == 0]
    print(f"  LLM judge OK. no-context mean={nz.mean():.3f} std={nz.std():.3f} n={len(nz)}")
except Exception as e:
    print(f"  LLM judge still failed: {e}")
    llm_ok = False

feature_cols = [c for c in train_df.columns if c not in
                ("context", "prompt_bn", "response_bn", "label", "task_type")]
y = train_df["label"].values

print("\n" + "=" * 60)
print("BASELINE (no llm_judge, no xlingual embed, no cultural_default,")
print("          no translation-based cross-lingual NLI)")
print("=" * 60)
drop_cols = ("llm_judge_score", "xlingual_consistency", "cultural_default_flag",
             "nli_en_entail", "nli_en_contra", "cross_lingual_disagreement")
base_cols = [c for c in feature_cols if c not in drop_cols]
cross_validate(train_df[base_cols].values, y)

print("\n" + "=" * 60)
print("+ translation-based cross-lingual NLI only")
print("=" * 60)
xlingual_cols = base_cols + ["nli_en_entail", "nli_en_contra", "cross_lingual_disagreement"]
cross_validate(train_df[xlingual_cols].values, y)

print("\n" + "=" * 60)
print(f"FULL FEATURE SET ({len(feature_cols)} features, llm_ok={llm_ok})")
print("=" * 60)
cross_validate(train_df[feature_cols].values, y)

print("\n" + "=" * 60)
print("Feature importance (full model on all 299 rows)")
print("=" * 60)
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
)
model.fit(train_df[feature_cols].values, y)
for col, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    print(f"   {col:30s}: {imp:.4f}")

train_df.to_pickle("fast_cv_train_df.pkl")
print("\nUpdated pickle with real llm_judge_score.")
