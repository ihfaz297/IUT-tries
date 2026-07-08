"""
Unified Joggota Submission Pipeline
Merges tithy-4.ipynb (NLI + XGBoost) with joggota_core.py (Form Engine + Deterministic Rules)
+ TigerLLM Judge for No-Context Rows
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig

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
# 2. MODELS (Embeddings, NLI & LLM Judge)
# ----------------------------------------------------------------------
class Embedder:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        # Run on CPU to save GPU VRAM for TigerLLM-9B
        self.model = SentenceTransformer(model_name, device="cpu")

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
    def __init__(self, model_name=NLI_MODEL_NAME):
        # Run on CPU to save GPU VRAM for TigerLLM-9B
        self.device = "cpu"
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

class TigerLLMJudge:
    def __init__(self, model_name=TIGERLLM_MODEL_NAME):
        print(f"Loading TigerLLM judge ({model_name}) in 4-bit...")
        self.tk = AutoTokenizer.from_pretrained(model_name)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_use_double_quant=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto"
        ).eval()
        self.dev = next(self.llm.parameters()).device
        
        def digit_ids(d):
            ids = set()
            for s in (d, " "+d):
                e = self.tk.encode(s, add_special_tokens=False)
                if e: ids.add(e[-1])
            return list(ids)
            
        self.ids1 = digit_ids("1")
        self.ids0 = digit_ids("0")

    def get_category(self, prompt_text, ctx_text, response_text):
        p_lower = str(prompt_text).lower()
        combined_text = f"{prompt_text} {ctx_text} {response_text}"
        
        if re.search(r'[a-zA-Z]', combined_text):
            return "code_mixed"
        elif ctx_text and str(ctx_text).strip() and str(ctx_text).strip() != "[NULL]":
            return "comprehension"
        elif any(k in p_lower for k in ["অর্থ", "ভাবার্থ", "সমার্থক", "বিপরীত", "মানে কী"]):
            return "vocabulary"
        elif any(c.isdigit() for c in p_lower):
            return "math"
        else:
            return "general_knowledge"

    def build_sys_prompt(self, category):
        base_rule = "কোনো ব্যাখ্যা দেবেন না। শুধুমাত্র একটি সংখ্যা আউটপুট দিন।"
        
        if category == "code_mixed":
            return (f"আপনি একজন বহুভাষিক (Multilingual) এবং কোড-মিক্সড (Code-Mixed/Banglish) ডেটা বিশ্লেষক। "
                    f"এখানে প্রশ্ন, অনুচ্ছেদ বা উত্তরে বাংলা, ইংরেজি বা বাংলিশ ভাষার মিশ্রণ থাকতে পারে। "
                    f"ভাষার মিশ্রণ থাকা সত্ত্বেও মূল অর্থটি যাচাই করুন। প্রস্তাবিত উত্তরটি যদি সঠিক এবং প্রাসঙ্গিক হয় (Correct), তবে শুধুমাত্র '1' লিখুন। "
                    f"উত্তরটি যদি ভুল, বানোয়াট বা অপ্রাসঙ্গিক হয় (Hallucinated), তবে শুধুমাত্র '0' লিখুন। {base_rule}")
        elif category == "vocabulary":
            return (f"আপনি একজন বিশেষজ্ঞ বাংলা ভাষাবিদ। আপনার কাজ হলো প্রশ্নের প্রদত্ত শব্দ বা বাগধারার অর্থ যাচাই করা। "
                    f"প্রস্তাবিত উত্তরটি যদি শব্দ বা বাগধারার সঠিক অর্থ বা ভাবার্থ প্রকাশ করে (Correct), তবে শুধুমাত্র '1' লিখুন। "
                    f"উত্তরটি যদি ভুল বা মনগড়া হয় (Hallucinated), তবে শুধুমাত্র '0' লিখুন। {base_rule}")
        elif category == "math":
            return (f"আপনি একজন গাণিতিক মূল্যায়নকারী এআই। প্রস্তাবিত গাণিতিক উত্তরটি নিখুঁত হিসাব-নিকাশ মেনে চলে কি ছুটি যাচাই করুন। "
                    f"হিসাব ও চূড়ান্ত উত্তর সম্পূর্ণ সঠিক (Correct) হলে শুধুমাত্র '1' লিখুন। "
                    f"হিসাবে সামান্যতম ভুল বা গরমিল (Wrong) থাকলে শুধুমাত্র '0' লিখুন। {base_rule}")
        else: # general_knowledge, history, literature
            return (f"আপনি একজন অত্যন্ত কঠোর তথ্য-যাচাইকারী (Fact-checker)। আপনার কাজ হলো সাধারণ জ্ঞান, বাঙালি সাহিত্য, এবং ঐতিহাসিক তথ্যের সত্যতা যাচাই করা। "
                    f"সতর্কতা: কোনো বিখ্যাত উক্তি, বইয়ের নাম বা লেখকের নাম (যেমন হুমায়ূন আহমেদ, তসলিমা নাসরিন, রবীন্দ্রনাথ ইত্যাদি) সম্পর্কে প্রচলিত স্টেরিওটাইপ দ্বারা প্রভাবিত হবেন না। "
                    f"প্রস্তাবিত উত্তরটি যদি বাস্তবিকভাবে সম্পূর্ণ সত্য এবং নির্ভুল হয় (Correct), তবে শুধুমাত্র '1' লিখুন। "
                    f"উত্তরটি যদি ভুল, মিথ্যা, বানোয়াট হয় বা ভুল লেখকের নাম উল্লেখ করে (Hallucinated), তবে শুধুমাত্র '0' লিখুন। {base_rule}")

    def score_no_context_rows(self, df):
        out = np.zeros(len(df))
        missing_mask = (df["has_context"] == 0)
        idx_to_score = df.index[missing_mask].tolist()
        
        if len(idx_to_score) == 0:
            return out

        print(f"Scoring {len(idx_to_score)} no-context rows with TigerLLM...")
        
        for count, i in enumerate(idx_to_score):
            r = df.loc[i]
            ctx = r.get("context", "")
            if pd.isna(ctx): ctx = ""
            
            cat = self.get_category(r["prompt_bn"], ctx, r["response_bn"])
            SYS = self.build_sys_prompt(cat)
            
            u = f"QUESTION: {r['prompt_bn']}\nANSWER: {r['response_bn']}\nVerdict:"
            
            enc = self.tk.apply_chat_template(
                [{"role":"system", "content":SYS}, {"role":"user", "content":u}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
            ii = enc["input_ids"][:, -768:].to(self.dev)
            am = enc["attention_mask"][:, -768:].to(self.dev)
            
            with torch.no_grad():
                lg = self.llm(input_ids=ii, attention_mask=am).logits[0, -1, :].float()
            
            p1 = torch.logsumexp(lg[self.ids1], 0)
            p0 = torch.logsumexp(lg[self.ids0], 0)
            
            # Softmax to get probability of generating '1' (Faithful)
            prob_faithful = torch.softmax(torch.stack([p0, p1]), 0)[1].item()
            out[i] = prob_faithful
            
            if count % 50 == 0 and count > 0: 
                print(f"  TigerLLM processed {count}/{len(idx_to_score)} no-context rows")
                
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

def _save_run_log(train_df, test_df, feature_cols, cv_results=None):
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
    }
    # Add feature means
    for col in feature_cols:
        if col in train_df.columns:
            row[f"train_{col}_mean"] = float(train_df[col].mean())
            row[f"train_{col}_std"] = float(train_df[col].std())
    # Cross-val F1 if available
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
    
    # ---- Dataset overview ----
    _log_dataset_stats(train_df, "TRAIN SET")
    _log_dataset_stats(test_df, "TEST SET")
    
    # --- PHASE 1: NLI & Embeddings ---
    print("\nLoading NLI & Embedding models...")
    embedder = Embedder()
    nli_scorer = NLIScorer()
    
    print("\n--- PHASE 1: Base Features (Form Engine + NLI + Embeddings) ---")
    print("--- Processing Train Set ---")
    train_df, feature_cols = build_base_features(train_df, embedder, nli_scorer)
    
    print("\n--- Processing Test Set ---")
    test_df, _ = build_base_features(test_df, embedder, nli_scorer)
    
    # ---- Task distribution ----
    _log_task_distribution(train_df)
    _log_task_distribution(test_df)
    
    # Wipe GPU memory clean before LLM
    print("\nCleaning up NLI/Embedder VRAM...")
    del embedder, nli_scorer
    gc.collect()
    torch.cuda.empty_cache()

    # --- PHASE 2: LLM Judge (Only for missing context) ---
    print("\n--- PHASE 2: TigerLLM Judge (No-Context Rows Only) ---")
    try:
        tiger_judge = TigerLLMJudge()
        train_df["tigerllm_faithful_prob"] = tiger_judge.score_no_context_rows(train_df)
        test_df["tigerllm_faithful_prob"] = tiger_judge.score_no_context_rows(test_df)
        feature_cols.append("tigerllm_faithful_prob")
        
        # Summary of TigerLLM scores
        train_llm = train_df["tigerllm_faithful_prob"]
        test_llm = test_df["tigerllm_faithful_prob"]
        print(f"\n🐯 TigerLLM Score Summary:")
        print(f"   Train: mean={train_llm.mean():.3f} | rows scored={train_llm[train_llm>0].count()}")
        print(f"   Test:  mean={test_llm.mean():.3f} | rows scored={test_llm[test_llm>0].count()}")
        
        # Cleanup LLM
        del tiger_judge
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ Warning: Failed to run TigerLLM judge: {e}")
        train_df["tigerllm_faithful_prob"] = 0
        test_df["tigerllm_faithful_prob"] = 0
        feature_cols.append("tigerllm_faithful_prob")

    # ---- Feature summary ----
    combined = pd.concat([train_df, test_df], ignore_index=True)
    _log_feature_summary(combined, feature_cols)

    # --- PHASE 3: ML Model ---
    print("\n--- PHASE 3: XGBoost Training & Cross-Validation ---")
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    
    # Cross-validation on training set
    print("\n📐 5-Fold Cross-Validation:")
    cv_scores = _cross_validate(X_train, y_train, n_folds=N_FOLDS)
    
    # Final train on full training set
    print("\n🏋️  Training final model on full training set...")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    # Feature importance
    importance = model.feature_importances_
    print("\n🔍 Feature Importance:")
    for col, imp in sorted(zip(feature_cols, importance), key=lambda x: -x[1]):
        print(f"   {col:30s}: {imp:.4f}")
    
    # Predict
    print("\nPredicting on test set...")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    _log_final_distribution(preds, test_df)
    
    # Save submission
    submission = pd.DataFrame({"id": test_df["id"].values, "label": preds})
    submission.to_csv(SUBMISSION_OUT, index=False)
    print(f"\n✅ Saved submission → {SUBMISSION_OUT}")
    
    # Save run log
    _save_run_log(train_df, test_df, feature_cols, cv_scores)
    print(f"✅ Run complete. Open {RUN_LOG_PATH} to compare across runs!")

if __name__ == "__main__":
    train_and_predict()
