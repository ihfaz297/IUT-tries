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
        "word_entropy", "char_entropy", "novel_char_ratio", "length_ratio", "deterministic_joggota"
    ]
    df[feature_cols] = df[feature_cols].fillna(0)
    
    return df, feature_cols

# ----------------------------------------------------------------------
# 5. XGBOOST TRAINING & INFERENCE
# ----------------------------------------------------------------------
def train_and_predict():
    print("Loading data...")
    train_df = load_json(TRAIN_PATH)
    test_df = load_test_csv(TEST_PATH)
    
    # --- PHASE 1: NLI & Embeddings ---
    print("Loading NLI & Embedding models...")
    embedder = Embedder()
    nli_scorer = NLIScorer()
    
    print("--- Processing Base Features (Train Set) ---")
    train_df, feature_cols = build_base_features(train_df, embedder, nli_scorer)
    
    print("--- Processing Base Features (Test Set) ---")
    test_df, _ = build_base_features(test_df, embedder, nli_scorer)
    
    # Wipe GPU memory clean before LLM
    print("Cleaning up NLI/Embedder VRAM...")
    del embedder, nli_scorer
    gc.collect()
    torch.cuda.empty_cache()

    # --- PHASE 2: LLM Judge (Only for missing context) ---
    print("--- Loading LLM Judge ---")
    try:
        tiger_judge = TigerLLMJudge()
        train_df["tigerllm_faithful_prob"] = tiger_judge.score_no_context_rows(train_df)
        test_df["tigerllm_faithful_prob"] = tiger_judge.score_no_context_rows(test_df)
        feature_cols.append("tigerllm_faithful_prob")
        
        # Cleanup LLM
        del tiger_judge
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to run TigerLLM judge (Missing GPU/Memory?): {e}")
        train_df["tigerllm_faithful_prob"] = 0
        test_df["tigerllm_faithful_prob"] = 0
        feature_cols.append("tigerllm_faithful_prob")

    # --- PHASE 3: ML Model ---
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
    probs = model.predict_proba(X_test)[:, 1]
    
    preds = (probs >= 0.5).astype(int)
    
    submission = pd.DataFrame({"id": test_df["id"].values, "label": preds})
    submission.to_csv(SUBMISSION_OUT, index=False)
    print(f"Success! Saved {SUBMISSION_OUT}")

if __name__ == "__main__":
    train_and_predict()
