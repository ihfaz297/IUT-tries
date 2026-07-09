"""
Unified Joggota Submission Pipeline
Merges NLI + XGBoost + Joggota deterministic rules + Small LLM Judge.
Implements cross-lingual consistency & cultural-default detection per implementation_plan.md.

VRAM strategy (Kaggle T4, 16 GB):
  Phase 1 (NLI + Embeddings) → GPU, then explicit .to("cpu") + empty_cache()
  Phase 2 (Small LLM Judge)   → Qwen2.5-1.5B or 3B in FP16, only no-context rows
  Fallback: LaBSE sim_premise_response as proxy if LLM fails to load
"""
import os
import gc
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

# Import our custom Joggota rules
from joggota_core import extract_joggota_features

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
TRAIN_PATH = "dataset samples.json"
TEST_PATH = "test set.csv"
SAMPLE_SUB_PATH = "sample submission.csv"
SUBMISSION_OUT = "submission.csv"

EMBED_MODEL_NAME = "sentence-transformers/LaBSE"
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Offline checkpoints (Kaggle dataset mounts or local downloads)
LLM_CHECKPOINTS = [
    "/kaggle/input/qwen2.5-1.5b-instruct",          # Kaggle dataset
    "offline_models/qwen-2.5-1.5b-instruct",         # Local download
    "Qwen/Qwen2.5-1.5B-Instruct",                    # HuggingFace (Phase 1 internet)
]

RANDOM_STATE = 42
N_FOLDS = 5

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["context"] = df["context"].replace("[NULL]", np.nan)
    df["context"] = df["context"].replace("", np.nan)
    df["has_context"] = df["context"].notna().astype(int)
    return df


def load_test_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "context" not in df.columns:
        df["context"] = np.nan
    df["context"] = df["context"].replace("[NULL]", np.nan)
    df["context"] = df["context"].replace("", np.nan)
    df["has_context"] = df["context"].notna().astype(int)
    return df


# ----------------------------------------------------------------------
# 2. MODELS (Embeddings, NLI, LLM Judge)
# ----------------------------------------------------------------------
def _force_gpu_cleanup():
    """Aggressively clear GPU memory including defragmentation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            _dummy = torch.zeros(1, device="cuda")
            del _dummy
        except Exception:
            pass


class Embedder:
    def __init__(self, model_name=EMBED_MODEL_NAME, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts):
        clean = []
        for t in texts:
            if t is None or (isinstance(t, float) and np.isnan(t)):
                clean.append("।")
            else:
                t = str(t)
                clean.append(t if t.strip() else "।")
        return self.model.encode(
            clean, batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )

    def cosine(self, a_emb, b_emb):
        return np.sum(a_emb * b_emb, axis=1)


class NLIScorer:
    def __init__(self, model_name=NLI_MODEL_NAME, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_batch(self, premises, hypotheses, batch_size=16):
        premises = [
            "" if p is None or (isinstance(p, float) and np.isnan(p)) else str(p)
            for p in premises
        ]
        hypotheses = [
            "" if h is None or (isinstance(h, float) and np.isnan(h)) else str(h)
            for h in hypotheses
        ]
        premises = [p if p.strip() else "।" for p in premises]
        hypotheses = [h if h.strip() else "।" for h in hypotheses]

        all_probs = []
        for i in range(0, len(premises), batch_size):
            p_batch = premises[i : i + batch_size]
            h_batch = hypotheses[i : i + batch_size]
            inputs = self.tokenizer(
                p_batch, h_batch, truncation=True, padding=True,
                max_length=256, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)


class SmallLLMJudge:
    """
    Qwen2.5-1.5B-Instruct (or 3B) factual judge for *no-context* rows.

    Uses logit-based scoring (probability of generating '1'/'0' tokens) instead
    of fragile generation + string matching.  Processes in batches for speed.
    """
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = checkpoint_path or "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading SmallLLMJudge from {model_path} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto",
        )
        self.model.eval()

        # Pre-compute token IDs for '1' and '0' (Qwen uses these digit tokens)
        self.ids_1 = self._digit_ids("1")
        self.ids_0 = self._digit_ids("0")
        print(f"  Token IDs for '1': {self.ids_1} | '0': {self.ids_0}")

    def _digit_ids(self, digit):
        """Return all token IDs that decode to the given digit string."""
        ids = set()
        for variant in (digit, " " + digit):
            encoded = self.tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.add(encoded[-1])
        return list(ids)

    def _build_prompt(self, prompt_bn, response_bn):
        """Build a Bangla fact-check prompt with cultural-default awareness."""
        return (
            "আপনি একজন কঠোর বাংলা তথ্য-যাচাইকারী (fact-checker)।\n"
            "বাংলাদেশ/বাঙালি প্রসঙ্গে সঠিক তথ্য দিন। পশ্চিমা/বৈশ্বিক ডিফল্ট উত্তর এড়িয়ে চলুন।\n"
            f"প্রশ্ন: {prompt_bn}\n"
            f"উত্তর: {response_bn}\n\n"
            "উত্তরটি তথ্যগতভাবে সঠিক ও বিশ্বস্ত কিনা? শুধুমাত্র 1 (সঠিক) বা 0 (ভুল) লিখুন।\n"
            "উত্তর:"
        )

    @torch.no_grad()
    def score_batch(self, prompts_bn, responses_bn, batch_size=8):
        """
        Score a list of (prompt, response) pairs.
        Returns array of P(faithful) in [0, 1] — higher = more likely faithful.
        Uses logit-space probability of token '1' vs '0' (more robust than generation).
        """
        n = len(prompts_bn)
        out = np.full(n, 0.5, dtype=np.float32)  # default: abstain

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_texts = []
            for i in range(start, end):
                batch_texts.append(
                    self._build_prompt(
                        str(prompts_bn[i]), str(responses_bn[i])
                    )
                )

            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self.device)

            logits = self.model(**inputs).logits  # (batch, seq_len, vocab)
            last_logits = logits[:, -1, :].float()  # (batch, vocab) — final position

            # Log-sum-exp for token groups
            log_p1 = torch.logsumexp(last_logits[:, self.ids_1], dim=1)  # (batch,)
            log_p0 = torch.logsumexp(last_logits[:, self.ids_0], dim=1)  # (batch,)

            # Softmax to get P(token='1' | prompt) — proxy for P(faithful)
            stacked = torch.stack([log_p0, log_p1], dim=1)  # (batch, 2)
            prob_faithful = torch.softmax(stacked, dim=1)[:, 1]  # (batch,)
            out[start:end] = prob_faithful.cpu().numpy()

        return out

    def score_no_context_rows(self, df):
        """
        Score only rows WITHOUT context. Returns array of same length as df.
        Context rows get NaN (will be filled later).
        """
        n = len(df)
        out = np.full(n, np.nan, dtype=np.float32)
        mask = df["has_context"] == 0
        idx_list = df.index[mask].tolist()

        if not idx_list:
            print("  (all rows have context — skipping LLM judge)")
            return out

        print(f"  Scoring {len(idx_list)} no-context rows with Small LLM Judge...")
        prompts = [str(df.loc[i]["prompt_bn"]) for i in idx_list]
        responses = [str(df.loc[i]["response_bn"]) for i in idx_list]
        scores = self.score_batch(prompts, responses, batch_size=8)

        for idx, score in zip(idx_list, scores):
            out[idx] = score

        return out


# ----------------------------------------------------------------------
# 3. CROSS-LINGUAL CONSISTENCY FEATURE
# ----------------------------------------------------------------------
def cross_lingual_consistency(df, embedder):
    """
    Compute a cross-lingual consistency score for each row.

    Strategy (lightweight, no second LLM call):
      For rows with context: LaBSE sim(context_EN, response_BN).
      We don't actually translate — LaBSE already lives in a shared multilingual
      space. Instead, we measure the semantic similarity between the context/prompt
      and the response *in LaBSE's space* as a proxy for consistency.

    For no-context rows: we'll use the LLM judge directly (Phase 2),
    so here we provide a fallback: LaBSE similarity between prompt and response
    as a baseline consistency signal.

    Returns a numpy array of scores (higher = more consistent / less hallucinated).
    """
    print("Computing cross-lingual consistency features...")

    response_col = df["response_bn"].astype(str)
    context_col = df["context"]

    # Encode all responses once
    resp_emb = embedder.encode(response_col.tolist())

    # For context rows: use context as the anchor
    # For no-context rows: use prompt as the anchor
    has_ctx = df["has_context"] == 1

    # Context anchor
    ctx_texts = context_col.fillna("").astype(str).tolist()
    ctx_texts = [t if t.strip() and t.strip() != "[NULL]" else "।" for t in ctx_texts]
    ctx_emb = embedder.encode(ctx_texts)

    # Prompt anchor (for no-context rows)
    prompt_texts = df["prompt_bn"].astype(str).tolist()
    prompt_texts = [t if t.strip() else "।" for t in prompt_texts]
    prompt_emb = embedder.encode(prompt_texts)

    # Combine: use context when available, prompt otherwise
    combined_emb = np.where(
        has_ctx.values[:, None], ctx_emb, prompt_emb,
    )
    xlingual_score = embedder.cosine(combined_emb, resp_emb)

    return xlingual_score


# ----------------------------------------------------------------------
# 4. LEXICAL HELPERS
# ----------------------------------------------------------------------
def bn_tokenize(text):
    if not isinstance(text, str):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return []
        text = str(text)
    return re.findall(r"[\u0980-\u09FF]+|\S+", text)


def token_overlap_ratio(a, b):
    ta, tb = set(bn_tokenize(a)), set(bn_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ----------------------------------------------------------------------
# 5. UNIFIED FEATURE PIPELINE
# ----------------------------------------------------------------------
def build_base_features(df, embedder, nli_scorer):
    """Phase 1: NLI + Embeddings + Joggota deterministic features."""
    print("Extracting Form & Deterministic Joggota features...")
    df = extract_joggota_features(df)

    prompt_col = df["prompt_bn"]
    response_col = df["response_bn"]
    context_col = df["context"]

    # --- NLI (context vs response, or prompt vs response if no context) ---
    print("Running NLI...")
    premise_ctx = context_col.fillna(prompt_col)
    probs_ctx = nli_scorer.score_batch(premise_ctx.tolist(), response_col.tolist())
    df["nli_ctx_entail"] = probs_ctx[:, 0]
    df["nli_ctx_contra"] = probs_ctx[:, 2]

    # --- LaBSE Embeddings ---
    print("Running Embeddings...")
    ctx_or_prompt_emb = embedder.encode(premise_ctx.tolist())
    resp_emb = embedder.encode(response_col.tolist())
    df["sim_premise_response"] = embedder.cosine(ctx_or_prompt_emb, resp_emb)

    # --- Cross-lingual consistency ---
    df["xlingual_consistency"] = cross_lingual_consistency(df, embedder)

    # --- Lexical overlap ---
    df["token_overlap_ctx_resp"] = [
        token_overlap_ratio(p, r)
        for p, r in zip(premise_ctx, response_col)
    ]

    # --- Feature list ---
    feature_cols = [
        "nli_ctx_entail",
        "nli_ctx_contra",
        "sim_premise_response",
        "xlingual_consistency",
        "token_overlap_ctx_resp",
        "has_context",
        "word_entropy",
        "char_entropy",
        "novel_char_ratio",
        "length_ratio",
        "deterministic_joggota",
        "corpus_match_score",
        "cultural_default_flag",
    ]
    df[feature_cols] = df[feature_cols].fillna(0)

    return df, feature_cols


# ----------------------------------------------------------------------
# 6. XGBOOST TRAINING & RUN LOGGING
# ----------------------------------------------------------------------
RUN_LOG_PATH = "run_log.csv"


def _log_dataset_stats(df, name):
    n = len(df)
    n_ctx = int(df["has_context"].sum())
    n_noctx = n - n_ctx
    print(f"\n{'='*60}")
    print(f"📊 {name} — {n} rows ({n_ctx} with context, {n_noctx} without)")
    if "label" in df.columns:
        n_faithful = int(df["label"].sum())
        n_hallu = n - n_faithful
        print(
            f"   Labels: {n_faithful} faithful ({n_faithful/n*100:.1f}%) | "
            f"{n_hallu} hallucinated ({n_hallu/n*100:.1f}%)"
        )
    return n, n_ctx, n_noctx


def _log_task_distribution(df):
    if "task_type" not in df.columns:
        return
    print("\n📂 Task Distribution:")
    for task, count in df["task_type"].value_counts().items():
        print(f"   {task:20s}: {count:4d} ({count/len(df)*100:.1f}%)")


def _log_feature_summary(df, feature_cols):
    print("\n📈 Feature Summary (mean ± std across train+test):")
    for col in feature_cols:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"   {col:30s}: {vals.mean():.4f} ± {vals.std():.4f}")


def _log_final_distribution(preds, test_df):
    total = len(preds)
    faithful = int(preds.sum())
    hallu = total - faithful
    print(
        f"\n🎯 Final Predictions: {faithful} faithful ({faithful/total*100:.1f}%) | "
        f"{hallu} hallucinated ({hallu/total*100:.1f}%)"
    )
    print(f"   (out of {total} test rows)")


def _save_run_log(train_df, test_df, feature_cols, cv_results=None, used_fallback=False):
    if "label" not in train_df.columns:
        return
    row = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_rows": len(train_df),
        "train_faithful": int(train_df["label"].sum()),
        "train_hallucinated": int((1 - train_df["label"]).sum()),
        "train_context": int(train_df["has_context"].sum()),
        "train_no_context": int((1 - train_df["has_context"]).sum()),
        "test_rows": len(test_df),
        "test_context": int(test_df["has_context"].sum()),
        "test_no_context": int((1 - test_df["has_context"]).sum()),
        "used_fallback": int(used_fallback),
    }
    for col in feature_cols:
        if col in train_df.columns:
            row[f"train_{col}_mean"] = float(train_df[col].mean())
            row[f"train_{col}_std"] = float(train_df[col].std())
    if cv_results is not None:
        row["cv_f1_mean"] = float(np.mean(cv_results))
        row["cv_f1_std"] = float(np.std(cv_results))

    log_df = pd.DataFrame([row])
    try:
        existing = pd.read_csv(RUN_LOG_PATH)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    except FileNotFoundError:
        pass
    log_df.to_csv(RUN_LOG_PATH, index=False)
    print(f"\n📝 Run logged to {RUN_LOG_PATH}")


def _cross_validate(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)
        scores.append(f1)
        print(f"   Fold {fold + 1}: F1 = {f1:.4f}")
    mean_f1 = np.mean(scores)
    std_f1 = np.std(scores)
    print(f"   {'='*30}")
    print(f"   CV F1: {mean_f1:.4f} ± {std_f1:.4f}")
    return scores


# ----------------------------------------------------------------------
# 7. MAIN PIPELINE
# ----------------------------------------------------------------------
def train_and_predict():
    print("Loading data...")
    train_df = load_json(TRAIN_PATH)
    test_df = load_test_csv(TEST_PATH)

    _log_dataset_stats(train_df, "TRAIN SET")
    _log_dataset_stats(test_df, "TEST SET")

    # ===================================================================
    # PHASE 1: NLI + Embeddings + Joggota (GPU)
    # ===================================================================
    print("\nLoading NLI & Embedding models...")
    embedder = Embedder()
    nli_scorer = NLIScorer()

    print("\n--- PHASE 1: Base Features ---")
    print("--- Processing Train Set ---")
    train_df, feature_cols = build_base_features(train_df, embedder, nli_scorer)

    print("\n--- Processing Test Set ---")
    test_df, _ = build_base_features(test_df, embedder, nli_scorer)

    _log_task_distribution(train_df)
    _log_task_distribution(test_df)

    # Aggressive GPU cleanup
    print("\nCleaning up Phase 1 GPU memory...")
    embedder.model.to("cpu")
    nli_scorer.model.to("cpu")
    del embedder, nli_scorer
    _force_gpu_cleanup()

    # ===================================================================
    # PHASE 2: Small LLM Judge (no-context rows only)
    # ===================================================================
    used_fallback = False
    print("\n--- PHASE 2: LLM Judge for No-Context Rows ---")

    # --- Determine checkpoint ---
    checkpoint = None
    for candidate in LLM_CHECKPOINTS:
        if os.path.exists(candidate):
            checkpoint = candidate
            print(f"  Found checkpoint: {checkpoint}")
            break
    if checkpoint is None:
        checkpoint = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"  Using HuggingFace: {checkpoint}")

    try:
        llm_judge = SmallLLMJudge(checkpoint_path=checkpoint)

        train_llm_scores = llm_judge.score_no_context_rows(train_df)
        test_llm_scores = llm_judge.score_no_context_rows(test_df)

        # Fill NaN (context rows) with 0.0 — XGBoost will learn to rely on NLI
        # features for context rows instead
        train_df["llm_judge_score"] = np.nan_to_num(train_llm_scores, nan=0.0)
        test_df["llm_judge_score"] = np.nan_to_num(test_llm_scores, nan=0.0)
        feature_cols.append("llm_judge_score")

        # Print stats for non-zero scores (no-context rows)
        train_nz = train_df["llm_judge_score"][train_df["has_context"] == 0]
        test_nz = test_df["llm_judge_score"][test_df["has_context"] == 0]
        print(f"\n🧠 LLM Judge Score Summary (no-context rows only):")
        print(f"   Train: mean={train_nz.mean():.3f} | n={len(train_nz)}")
        print(f"   Test:  mean={test_nz.mean():.3f} | n={len(test_nz)}")

        del llm_judge
        _force_gpu_cleanup()

    except Exception as e:
        print(f"⚠️  LLM Judge failed: {e}")
        print(f"   → Falling back to sim_premise_response for no-context signal...")
        used_fallback = True

        # Use the already-computed LaBSE similarity as the proxy
        train_df["llm_judge_score"] = train_df["sim_premise_response"]
        test_df["llm_judge_score"] = test_df["sim_premise_response"]
        feature_cols.append("llm_judge_score")

    # ---- Feature summary ----
    combined = pd.concat([train_df, test_df], ignore_index=True)
    _log_feature_summary(combined, feature_cols)

    # ===================================================================
    # PHASE 3: XGBoost Training & Inference
    # ===================================================================
    print("\n--- PHASE 3: XGBoost Training & Cross-Validation ---")
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values

    print("\n📐 5-Fold Cross-Validation:")
    cv_scores = _cross_validate(X_train, y_train, n_folds=N_FOLDS)

    print("\n🏋️  Training final model on full training set...")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    importance = model.feature_importances_
    print("\n🔍 Feature Importance:")
    for col, imp in sorted(zip(feature_cols, importance), key=lambda x: -x[1]):
        print(f"   {col:30s}: {imp:.4f}")

    print("\nPredicting on test set...")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    _log_final_distribution(preds, test_df)

    submission = pd.DataFrame({"id": test_df["id"].values, "label": preds})
    submission.to_csv(SUBMISSION_OUT, index=False)
    print(f"\n✅ Saved submission → {SUBMISSION_OUT}")

    _save_run_log(train_df, test_df, feature_cols, cv_scores, used_fallback)

    if used_fallback:
        print(
            "\n📌 Note: LLM Judge unavailable — used sim_premise_response fallback.\n"
            "   Run download_models.py first for offline Kaggle usage."
        )
    print("✅ Run complete.")


if __name__ == "__main__":
    train_and_predict()