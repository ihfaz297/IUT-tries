"""
Unified Joggota Submission Pipeline
Merges tithy-4.ipynb (NLI + XGBoost) with joggota_core.py (Form Engine + Deterministic Rules)
+ Fine-tuned BanglaBERT-large Classifier for All Rows

VRAM strategy:
  Phase 1 (NLI + Embeddings) → GPU, then explicit .to("cpu") + empty_cache()
  Phase 2 (BanglaBERT-large) → GPU, batch inference on ALL rows (context + no-context)
  Fallback: If BanglaBERT fails, use Phase 1 sim_premise_response as proxy signal
"""
import os
import gc
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

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

# On Kaggle, change this to your offline dataset path, e.g. "/kaggle/input/tigerllm-9b-it"
TIGERLLM_MODEL_NAME = "md-nishat-008/TigerLLM-9B-it"

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
# 2. MODELS (Embeddings, NLI, LLM Judge, Fallback)
# ----------------------------------------------------------------------
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
        return self.model.encode(clean, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

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
        premises = ["" if p is None or (isinstance(p, float) and np.isnan(p)) else str(p) for p in premises]
        hypotheses = ["" if h is None or (isinstance(h, float) and np.isnan(h)) else str(h) for h in hypotheses]
        premises = [p if p.strip() else "।" for p in premises]
        hypotheses = [h if h.strip() else "।" for h in hypotheses]

        all_probs = []
        for i in range(0, len(premises), batch_size):
            p_batch = premises[i:i + batch_size]
            h_batch = hypotheses[i:i + batch_size]
            inputs = self.tokenizer(p_batch, h_batch, truncation=True, padding=True, max_length=256, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

def _force_gpu_cleanup():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # Small allocation to defragment
    try:
        _dummy = torch.zeros(1, device="cuda")
        del _dummy
    except:
        pass

class PairDataset(Dataset):
    """Minimal (premise, response) dataset for BanglaBERT cross-encoder inference."""
    def __init__(self, premises, responses, tokenizer, max_len=384):
        self.premises = list(premises)
        self.responses = list(responses)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, i):
        enc = self.tokenizer(
            str(self.premises[i]), str(self.responses[i]),
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class BanglaBERTClassifier:
    """
    Fine-tuned BanglaBERT-large cross-encoder for faithfulness scoring.
    Loads from a local checkpoint if available, otherwise from HuggingFace.
    350M params — fits comfortably on T4 alongside other models.
    """
    BANGLA_MODEL = "csebuetnlp/banglabert_large"

    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = checkpoint_path or self.BANGLA_MODEL
        print(f"Loading BanglaBERT classifier from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2, ignore_mismatched_sizes=True
        ).half().eval().to(self.device)

    @torch.no_grad()
    def score_all_rows(self, df, batch_size=32):
        """
        Score ALL rows (context + no-context) in batches.
        Uses context-or-prompt as premise, response as hypothesis.
        Returns a numpy array of P(faithful) probabilities (0–1).
        """
        out = np.zeros(len(df))
        # Build premise: context if available, else prompt
        premises = []
        for _, r in df.iterrows():
            ctx = r.get("context", "")
            if pd.isna(ctx) or str(ctx).strip().lower() in ("[null]", "null", "none", "nan", ""):
                premises.append(str(r["prompt_bn"]))
            else:
                premises.append(str(r["prompt_bn"]) + " " + str(ctx).strip())
        responses = df["response_bn"].astype(str).tolist()

        ds = PairDataset(premises, responses, self.tokenizer, max_len=384)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

        all_probs = []
        for batch in loader:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device)
                ).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)

        probs = np.concatenate(all_probs)
        out[:] = probs
        return out


# ----------------------------------------------------------------------
# 3. LEXICAL HELPERS (From tithy-4)
# ----------------------------------------------------------------------
def bn_tokenize(text):
    if not isinstance(text, str):
        if text is None or (isinstance(text, float) and np.isnan(text)): return []
        text = str(text)
    return re.findall(r"[\u0980-\u09FF]+|\S+", text)

def token_overlap_ratio(a, b):
    ta, tb = set(bn_tokenize(a)), set(bn_tokenize(b))
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

# ----------------------------------------------------------------------
# 4. UNIFIED FEATURE PIPELINE
# ----------------------------------------------------------------------
def build_base_features(df, embedder, nli_scorer):
    print("Extracting Form & Deterministic Joggota features...")
    df = extract_joggota_features(df)
    
    prompt_col = df["prompt_bn"]
    response_col = df["response_bn"]
    context_col = df["context"]
    
    print("Running NLI...")
    premise_ctx = context_col.fillna(prompt_col)
    probs_ctx = nli_scorer.score_batch(premise_ctx.tolist(), response_col.tolist())
    df["nli_ctx_entail"] = probs_ctx[:, 0]
    df["nli_ctx_contra"] = probs_ctx[:, 2]

    print("Running Embeddings...")
    ctx_or_prompt_emb = embedder.encode(premise_ctx.tolist())
    resp_emb = embedder.encode(response_col.tolist())
    df["sim_premise_response"] = embedder.cosine(ctx_or_prompt_emb, resp_emb)
    
    df["token_overlap_ctx_resp"] = [token_overlap_ratio(p, r) for p, r in zip(premise_ctx, response_col)]

    feature_cols = [
        "nli_ctx_entail", "nli_ctx_contra", "sim_premise_response", "token_overlap_ctx_resp", "has_context",
        "word_entropy", "char_entropy", "novel_char_ratio", "length_ratio", "deterministic_joggota",
        "corpus_match_score"
    ]
    df[feature_cols] = df[feature_cols].fillna(0)
    
    return df, feature_cols

# ----------------------------------------------------------------------
# 5. XGBOOST TRAINING & INFERENCE
# ----------------------------------------------------------------------
# ==========================================
# 6. RUN LOGGER — tracks per-run statistics
# ==========================================
RUN_LOG_PATH = "run_log.csv"

def _log_dataset_stats(df, name):
    """Log basic dataset statistics."""
    n = len(df)
    n_ctx = df["has_context"].sum()
    n_noctx = n - n_ctx
    print(f"\n{'='*60}")
    print(f"📊 {name} — {n} rows ({n_ctx} with context, {n_noctx} without)")
    if "label" in df.columns:
        n_faithful = df["label"].sum()
        n_hallu = n - n_faithful
        print(f"   Labels: {n_faithful} faithful ({n_faithful/n*100:.1f}%) | {n_hallu} hallucinated ({n_hallu/n*100:.1f}%)")
    return n, n_ctx, n_noctx

def _log_task_distribution(df):
    """Print task type breakdown."""
    if "task_type" not in df.columns:
        return
    print("\n📂 Task Distribution:")
    for task, count in df["task_type"].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {task:20s}: {count:4d} ({pct:.1f}%)")

def _log_feature_summary(df, feature_cols):
    """Print mean/std for each numerical feature."""
    print("\n📈 Feature Summary (mean ± std across train+test):")
    for col in feature_cols:
        if col in df.columns:
            vals = df[col].dropna()
            mean = vals.mean()
            std = vals.std()
            print(f"   {col:30s}: {mean:.4f} ± {std:.4f}")

def _log_final_distribution(preds, test_df):
    """Print what the pipeline predicted."""
    total = len(preds)
    faithful = preds.sum()
    hallu = total - faithful
    print(f"\n🎯 Final Predictions: {faithful} faithful ({faithful/total*100:.1f}%) | {hallu} hallucinated ({hallu/total*100:.1f}%)")
    print(f"   (out of {total} test rows)")

def _save_run_log(train_df, test_df, feature_cols, cv_results=None, used_fallback=False):
    """Append a row to run_log.csv for git-pull insights."""
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
    """Run stratified k-fold CV and return per-fold F1 scores."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)
        scores.append(f1)
        print(f"   Fold {fold+1}: F1 = {f1:.4f}")
    mean_f1 = np.mean(scores)
    std_f1 = np.std(scores)
    print(f"   {'='*30}")
    print(f"   CV F1: {mean_f1:.4f} ± {std_f1:.4f}")
    return scores

def train_and_predict():
    print("Loading data...")
    train_df = load_json(TRAIN_PATH)
    test_df = load_test_csv(TEST_PATH)
    
    _log_dataset_stats(train_df, "TRAIN SET")
    _log_dataset_stats(test_df, "TEST SET")
    
    # --- PHASE 1: NLI & Embeddings (GPU) ---
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

    # --- PHASE 2: BanglaBERT Classifier (All Rows) ---
    used_fallback = False
    print("\n--- PHASE 2: BanglaBERT Faithfulness Scoring ---")
    
    try:
        # Require a fine-tuned checkpoint — base model has a random head
        checkpoint = None
        for candidate in (
            "/kaggle/working/banglabert_finetuned",          # Auto-trained by notebook Part 2
            "/kaggle/input/banglabert-finetuned-hallu",      # Pre-attached Kaggle dataset
            "banglabert_finetuned",                          # Local training output
            "banglabert_checkpoint.pt",                      # Legacy checkpoint
        ):
            if os.path.isdir(candidate) or (os.path.isfile(candidate) and candidate.endswith(".pt")):
                checkpoint = candidate
                break

        if checkpoint is None:
            raise FileNotFoundError(
                "No fine-tuned BanglaBERT checkpoint found. "
                "Skipping to sim_premise_response fallback — "
                "base banglabert_large has a random classification head "
                "and would inject noise into XGBoost."
            )

        banglabert = BanglaBERTClassifier(checkpoint_path=checkpoint)
        train_df["banglabert_faithful_prob"] = banglabert.score_all_rows(train_df, batch_size=32)
        test_df["banglabert_faithful_prob"] = banglabert.score_all_rows(test_df, batch_size=32)
        feature_cols.append("banglabert_faithful_prob")

        train_bb = train_df["banglabert_faithful_prob"]
        test_bb = test_df["banglabert_faithful_prob"]
        n_train_scored = (train_bb > 0).sum()
        n_test_scored = (test_bb > 0).sum()
        print(f"\n🧠 BanglaBERT Score Summary:")
        print(f"   Train: mean={train_bb.mean():.3f} | rows scored={n_train_scored}/{len(train_df)}")
        print(f"   Test:  mean={test_bb.mean():.3f} | rows scored={n_test_scored}/{len(test_df)}")

        del banglabert
        _force_gpu_cleanup()
    except Exception as e:
        print(f"⚠️ BanglaBERT failed: {e}")
        print(f"   → Falling back to Phase 1 sim_premise_response as proxy signal...")
        used_fallback = True

        # Use already-computed LaBSE prompt-response similarity from Phase 1
        # (no additional model load needed)
        train_df["banglabert_faithful_prob"] = train_df["sim_premise_response"]
        test_df["banglabert_faithful_prob"] = test_df["sim_premise_response"]
        feature_cols.append("banglabert_faithful_prob")

        train_bb = train_df["banglabert_faithful_prob"]
        test_bb = test_df["banglabert_faithful_prob"]
        print(f"\n🔤 sim_premise_response fallback summary:")
        print(f"   Train: mean={train_bb.mean():.3f}")
        print(f"   Test:  mean={test_bb.mean():.3f}")

    # ---- Feature summary ----
    combined = pd.concat([train_df, test_df], ignore_index=True)
    _log_feature_summary(combined, feature_cols)

    # --- PHASE 3: XGBoost ---
    print("\n--- PHASE 3: XGBoost Training & Cross-Validation ---")
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    
    print("\n📐 5-Fold Cross-Validation:")
    cv_scores = _cross_validate(X_train, y_train, n_folds=N_FOLDS)
    
    print("\n🏋️  Training final model on full training set...")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
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
        print(f"\n📌 Note: Used sim_premise_response fallback (BanglaBERT couldn't load)")
    print(f"✅ Run complete.")

if __name__ == "__main__":
    train_and_predict()