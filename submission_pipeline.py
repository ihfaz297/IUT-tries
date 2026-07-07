"""
Unified Joggota Submission Pipeline
Merges tithy-4.ipynb (NLI + XGBoost) with joggota_core.py (Form Engine + Deterministic Rules)
"""
import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb
import difflib
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
# 2. MODELS (Embeddings & NLI)
# ----------------------------------------------------------------------
class Embedder:
    def __init__(self, model_name=EMBED_MODEL_NAME, device=None):
        self.model = SentenceTransformer(model_name, device=device)

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
def build_features(df, embedder, nli_scorer):
    print("Extracting Form & Deterministic Joggota features...")
    df = extract_joggota_features(df) # Fuses joggota_core.py features
    
    prompt_col = df["prompt_bn"]
    response_col = df["response_bn"]
    context_col = df["context"]
    
    # NLI Features
    print("Running NLI...")
    premise_ctx = context_col.fillna(prompt_col)
    probs_ctx = nli_scorer.score_batch(premise_ctx.tolist(), response_col.tolist())
    df["nli_ctx_entail"] = probs_ctx[:, 0]
    df["nli_ctx_contra"] = probs_ctx[:, 2]

    # Embedding Features
    print("Running Embeddings...")
    ctx_or_prompt_emb = embedder.encode(premise_ctx.tolist())
    resp_emb = embedder.encode(response_col.tolist())
    df["sim_premise_response"] = embedder.cosine(ctx_or_prompt_emb, resp_emb)
    
    df["token_overlap_ctx_resp"] = [token_overlap_ratio(p, r) for p, r in zip(premise_ctx, response_col)]

    feature_cols = [
        # ML Features
        "nli_ctx_entail", "nli_ctx_contra", "sim_premise_response", "token_overlap_ctx_resp", "has_context",
        # Joggota Native Features
        "word_entropy", "char_entropy", "novel_char_ratio", "length_ratio", "deterministic_joggota"
    ]
    
    # Fill any remaining NaNs in features
    df[feature_cols] = df[feature_cols].fillna(0)
    
    return df, feature_cols

# ----------------------------------------------------------------------
# 5. XGBOOST TRAINING & INFERENCE
# ----------------------------------------------------------------------
def train_and_predict():
    print("Loading data...")
    train_df = load_json(TRAIN_PATH)
    test_df = load_test_csv(TEST_PATH)
    
    print("Loading models...")
    embedder = Embedder()
    nli_scorer = NLIScorer()
    
    print("--- Processing Train Set ---")
    train_df, feature_cols = build_features(train_df, embedder, nli_scorer)
    
    print("--- Processing Test Set ---")
    test_df, _ = build_features(test_df, embedder, nli_scorer)
    
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    print("Predicting...")
    # Basic threshold at 0.5 (Can tune later)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    submission = pd.DataFrame({"id": test_df["id"].values, "label": preds})
    submission.to_csv(SUBMISSION_OUT, index=False)
    print(f"Success! Saved {SUBMISSION_OUT}")

if __name__ == "__main__":
    train_and_predict()
