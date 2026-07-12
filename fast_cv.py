"""
Fast local CV harness — train-set-only (299 rows), CPU-friendly.
Does NOT touch the 2516-row test set, so it stays fast enough to iterate
on features/hyperparams locally before pushing a full run to Kaggle.

Usage: python fast_cv.py
"""
import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from submission_pipeline import (
    load_json, TRAIN_PATH, Embedder, NLIScorer, SmallLLMJudge, Translator,
    build_base_features, RANDOM_STATE, LLM_CHECKPOINTS, TRANSLATOR_CHECKPOINTS,
)

sys.stdout.reconfigure(encoding="utf-8")


def cross_validate(X, y, n_folds=5, max_depth=3, n_estimators=300, verbose=True):
    """
    label=1 is faithful, label=0 is hallucinated. Rules Section 7 states the
    primary metric is binary F1 on the HALLUCINATED class — sklearn's default
    f1_score(y, pred) uses pos_label=1 (faithful), which is the wrong class.
    Reports all three since Overview vs Rules disagree on macro vs binary-hallu.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    hallu_scores, faithful_scores, macro_scores = [], [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        )
        model.fit(X[tr_idx], y[tr_idx])
        preds = model.predict(X[val_idx])
        f1_hallu = f1_score(y[val_idx], preds, pos_label=0)
        f1_faithful = f1_score(y[val_idx], preds, pos_label=1)
        f1_macro = f1_score(y[val_idx], preds, average="macro")
        hallu_scores.append(f1_hallu)
        faithful_scores.append(f1_faithful)
        macro_scores.append(f1_macro)
        if verbose:
            print(f"   Fold {fold+1}: hallu_F1={f1_hallu:.4f}  faithful_F1={f1_faithful:.4f}  macro_F1={f1_macro:.4f}")
    print(f"   CV hallu_F1 (primary): {np.mean(hallu_scores):.4f} +/- {np.std(hallu_scores):.4f}")
    print(f"   CV faithful_F1:        {np.mean(faithful_scores):.4f} +/- {np.std(faithful_scores):.4f}")
    print(f"   CV macro_F1:           {np.mean(macro_scores):.4f} +/- {np.std(macro_scores):.4f}")
    return hallu_scores


def main():
    print("Loading train set...")
    train_df = load_json(TRAIN_PATH)
    print(f"  {len(train_df)} rows | {int(train_df['has_context'].sum())} ctx / "
          f"{int((1-train_df['has_context']).sum())} no-ctx")

    print("\nLoading NLI + Embedding + Translator models (CPU)...")
    embedder = Embedder(device="cpu")
    nli_scorer = NLIScorer(device="cpu")

    translator = None
    try:
        t_ckpt = next((c for c in TRANSLATOR_CHECKPOINTS if os.path.exists(c)),
                      "facebook/nllb-200-distilled-600M")
        translator = Translator(checkpoint_path=t_ckpt, device="cpu")
        xlingual_ok = True
    except Exception as e:
        print(f"Translator failed: {e}")
        xlingual_ok = False

    print("\nExtracting Phase 1 features (incl. cross-lingual translation — slow on CPU)...")
    train_df, feature_cols = build_base_features(train_df, embedder, nli_scorer, translator)

    print("\nLoading SmallLLMJudge (Qwen, CPU — this is the slow part)...")
    checkpoint = None
    for c in LLM_CHECKPOINTS:
        if os.path.exists(c):
            checkpoint = c
            break
    checkpoint = checkpoint or "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        judge = SmallLLMJudge(checkpoint_path=checkpoint, device="cpu")
        llm_scores = judge.score_no_context_rows(train_df)
        train_df["llm_judge_score"] = np.nan_to_num(llm_scores, nan=0.0)
        llm_ok = True
    except Exception as e:
        print(f"LLM judge failed: {e}")
        train_df["llm_judge_score"] = train_df["sim_premise_response"]
        llm_ok = False
    feature_cols = feature_cols + ["llm_judge_score"]

    y = train_df["label"].values

    print("\n" + "=" * 60)
    print("ABLATION: baseline (no llm_judge, no xlingual embed, no cultural_default,")
    print("          no translation-based cross-lingual NLI)")
    print("=" * 60)
    drop_cols = ("llm_judge_score", "xlingual_consistency", "cultural_default_flag",
                 "nli_en_entail", "nli_en_contra", "cross_lingual_disagreement")
    base_cols = [c for c in feature_cols if c not in drop_cols]
    cross_validate(train_df[base_cols].values, y)

    print("\n" + "=" * 60)
    print(f"+ translation-based cross-lingual NLI only (xlingual_ok={xlingual_ok})")
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
    print("\nSaved feature dataframe -> fast_cv_train_df.pkl (for reuse without recomputing)")


if __name__ == "__main__":
    main()
