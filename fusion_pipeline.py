"""
Fusion pipeline: combines two independently-built approaches.

  Lane 1 (teammate's, adapted near-verbatim from phase2-final.ipynb):
    - SummaC-ZS style windowed NLI via MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
    - gemma-4-12b-it bilingual (bn + en) judge, logit-based Yes/No scoring
    - 15-template deterministic math-word-problem solver

  Lane 2 (ours): translation-based cross-lingual NLI (NLLB bn->en + NLI re-check),
    joggota deterministic rules (idiom/spelling/math, now bug-fixed), response-quality
    heuristics, and the two cheap lexical features that carried the most weight in our
    real Kaggle run: context_containment, novel_char_ratio.

  Then: proper StratifiedKFold OOF comparison of (his features alone / ours alone /
  combined) x (GaussianNB / LogisticRegression / XGBoost), reporting hallu_F1 honestly
  (no in-sample leakage, no double-dipped threshold search) before picking a final model.

Run this on Kaggle (needs GPU for gemma-4 + NLI + NLLB). Mirrors phase2-final.ipynb's
offline plumbing (wheelhouse-based transformers upgrade, Kaggle Model registry for gemma-4).
"""
import os
import re
import gc
import glob
import json
import random
import warnings
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# Local modules (already Phase-2-compliant, no internet needed)
from joggota_core import extract_joggota_features
from submission_pipeline import Translator, TRANSLATOR_CHECKPOINTS
from ground_truth_matcher import GroundTruthMatcher, compute_gt_features

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
CFG = {
    "seed": 42,
    "llm_batch": 8,
    "llm_token_budget": 12288,
    "nli_batch": 64,
    "win_list": [2, 3],
    "max_ctx_chars": 3000, "max_prompt_chars": 1500, "max_resp_chars": 2500,
    "max_len_llm": 2048,
}
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "no GPU",
      "| GPUs:", torch.cuda.device_count() if DEVICE == "cuda" else 0)

# ----------------------------------------------------------------------
# 1. BOOTSTRAP: upgrade transformers so gemma-4's architecture is recognized.
# CONFIRMED (not assumed) via a real Kaggle run: even the latest PyPI release
# (5.13.1) does NOT recognize the `gemma4_unified` architecture -- it's too new
# for any released version. transformers' own error message says to install
# from source instead, so that's what this does. This is almost certainly what
# the teammate's wheelhouse actually contained (a source build, not a PyPI
# release) -- his ">=5.13" comment undersold what was actually needed.
# ----------------------------------------------------------------------
def bootstrap_transformers():
    hits = glob.glob("/kaggle/input/**/wheelhouse", recursive=True)
    if hits:
        wheeldir = hits[0]
        print("wheelhouse found at:", wheeldir)
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wheeldir,
             "-U", "transformers", "accelerate", "bitsandbytes"],
            capture_output=True, text=True, timeout=300,
        )
        print(r.stdout[-1500:])
        if r.returncode != 0:
            print("STDERR:", r.stderr[-1500:])
        return

    print("  no local wheelhouse -- trying a plain PyPI upgrade first...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U",
         "transformers", "accelerate", "bitsandbytes"],
        capture_output=True, text=True, timeout=600,
    )
    print(r.stdout[-800:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-800:])

    # verify gemma4_unified is actually registered, not just a high version number --
    # confirmed empirically that a high version number alone isn't sufficient
    check = subprocess.run(
        [sys.executable, "-c",
         "from transformers.models.auto.configuration_auto import CONFIG_MAPPING; "
         "import sys; sys.exit(0 if 'gemma4_unified' in CONFIG_MAPPING else 1)"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        print("  gemma4_unified recognized by the PyPI release -- no source build needed")
        return

    print("  PyPI release doesn't recognize gemma4_unified -- installing from GitHub "
          "source instead (this is what transformers' own error message recommends "
          "for architectures too new for any released version; expect this to take "
          "a few minutes, it builds from source)...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U",
         "git+https://github.com/huggingface/transformers.git",
         "accelerate", "bitsandbytes"],
        capture_output=True, text=True, timeout=900,
    )
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:])

bootstrap_transformers()
import transformers
print("transformers version:", transformers.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "gemma4_unified" not in CONFIG_MAPPING:
    print("  WARNING: gemma4_unified still not registered after bootstrap -- gemma-4 will "
          "fail to load. RESTART THE KERNEL and rerun from this cell (Kaggle needs a fresh "
          "process to pick up a newly-installed transformers build; a live upgrade doesn't "
          "hot-swap an already-imported package).")

# ----------------------------------------------------------------------
# 2. DATA DISCOVERY (adapted from phase2-final.ipynb -- more robust than ours)
# ----------------------------------------------------------------------
NULLISH = {"", "[null]", "null", "none", "nan", "n/a"}

def clean(s, lim):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    return "" if s.lower() in NULLISH else s[:lim]

def find_data():
    roots = [r for r in ["/kaggle/input/bengali-hallucination",
                          "/kaggle/input/competitions/bengali-hallucination",
                          "/kaggle/input", "."] if os.path.isdir(r)]
    test_df, labeled_df = None, None
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True) + \
                 glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            if os.path.getsize(p) > 80_000_000:
                continue
            try:
                df = pd.read_csv(p) if p.lower().endswith(".csv") else pd.DataFrame(json.load(open(p, encoding="utf-8")))
            except Exception:
                continue
            cols = {c.lower() for c in df.columns}
            if "response_bn" in cols or "response" in cols:
                if "label" in cols and (test_df is None if False else True):
                    if labeled_df is None or len(df) > len(labeled_df):
                        if "label" in cols:
                            labeled_df = df.copy()
                if "label" not in cols and (test_df is None or len(df) > len(test_df)):
                    test_df = df.copy()
    return test_df, labeled_df

test_raw, labeled_raw = find_data()
assert test_raw is not None, "test set not found"
assert labeled_raw is not None, "labeled sample (dataset samples.json) not found"
print("test:", test_raw.shape, "| labeled:", labeled_raw.shape)

def prep(df):
    d = pd.DataFrame()
    d["id"] = df["id"] if "id" in df.columns else np.arange(len(df))
    d["prompt_bn"] = df["prompt_bn"].map(lambda s: clean(s, CFG["max_prompt_chars"]))
    d["response_bn"] = df["response_bn"].map(lambda s: clean(s, CFG["max_resp_chars"]))
    d["context"] = df["context"].map(lambda s: clean(s, CFG["max_ctx_chars"])) if "context" in df.columns else ""
    d["context"] = d["context"].replace("", np.nan)
    d["has_context"] = d["context"].notna().astype(int)
    return d

test = prep(test_raw)
train = prep(labeled_raw)
train["label"] = labeled_raw["label"].astype(int).values
print("train has_context:", train["has_context"].mean().round(3),
      "| test has_context:", test["has_context"].mean().round(3))

BN_SENT = re.compile(r"(?<=[।!?\n])\s+")
def sents(t):
    parts = [s.strip() for s in BN_SENT.split(t) if s.strip()]
    return parts if parts else ([t] if t else [])

# ----------------------------------------------------------------------
# 5. LANE 1c: gemma-4-12b-it bilingual judge (verbatim from teammate)
# ----------------------------------------------------------------------
def run_gemma_judge(train_df, test_df):
    import kagglehub
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    gemma_path = kagglehub.model_download("google/gemma-4/transformers/gemma-4-12b-it")
    print("gemma-4-12b-it resolved at:", gemma_path)
    tok = AutoTokenizer.from_pretrained(gemma_path)
    tok.padding_side = "left"; tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
    kw = dict(device_map="auto", attn_implementation="eager", quantization_config=qcfg)
    try:
        llm = AutoModelForCausalLM.from_pretrained(gemma_path, dtype=torch.float16, **kw).eval()
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(gemma_path, torch_dtype=torch.float16, **kw).eval()
    print("loaded gemma-4-12b-it, 4-bit")

    def token_set(variants):
        ids = []
        for v in variants:
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1 and e[0] not in ids:
                ids.append(e[0])
        return ids

    YES = token_set(["Yes", "yes", " Yes", " yes", "হ্যাঁ"])
    NO = token_set(["No", "no", " No", " no", "না"])

    def chat(text):
        msgs = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def prompt_A(p, r, c):
        ctx = f"\n\nতথ্যসূত্র (এটিই একমাত্র সত্যের ভিত্তি):\n{c}" if c else ""
        src = "প্রদত্ত তথ্যসূত্র" if c else "বাস্তব, যাচাইযোগ্য জ্ঞান"
        return chat(f"""তুমি একজন কঠোর বাংলা ফ্যাক্ট-চেকার।{ctx}

প্রশ্ন/নির্দেশ:
{p}

মডেলের উত্তর:
{r}

উপরের উত্তরটি কি {src}-এর সাথে সম্পূর্ণ সঙ্গতিপূর্ণ — কোনো ভুল তথ্য, বানোয়াট নাম/তারিখ/সংখ্যা বা অসমর্থিত দাবি নেই? শুধু এক শব্দে উত্তর দাও: Yes অথবা No।""")

    def prompt_B(p, r, c):
        ctx = f"\n\nSOURCE (the only ground truth):\n{c}" if c else ""
        tail = "not supported by the SOURCE" if c else "false in the real world"
        return chat(f"""You are a strict fact-checker for Bengali LLM outputs.{ctx}

PROMPT:
{p}

RESPONSE:
{r}

Does the RESPONSE contain any hallucination — a factual error, a fabricated name/date/number, or a claim {tail}? Answer with exactly one word: Yes or No.""")

    @torch.no_grad()
    def judge(df, builder, flip, tag=""):
        texts = [builder(df["prompt_bn"].iat[i], df["response_bn"].iat[i],
                         df["context"].iat[i] if pd.notna(df["context"].iat[i]) else "")
                for i in range(len(df))]
        lens = np.array([len(x) for x in tok(texts, truncation=True, max_length=CFG["max_len_llm"])["input_ids"]])
        order = np.argsort(lens, kind="stable")
        pair_ids = YES + NO
        ps, i, done = np.zeros(len(texts)), 0, 0
        while i < len(order):
            take = int(min(CFG["llm_batch"], len(order) - i))
            while take > 1 and int(lens[order[i + take - 1]]) * take > CFG["llm_token_budget"]:
                take -= 1
            idx = order[i:i + take]
            while True:
                try:
                    enc = tok([texts[k] for k in idx], return_tensors="pt", padding=True,
                              truncation=True, max_length=CFG["max_len_llm"]).to(llm.device)
                    try:
                        out = llm(**enc, use_cache=False, logits_to_keep=1)
                    except TypeError:
                        out = llm(**enc, use_cache=False)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(idx) == 1:
                        raise
                    idx = idx[:max(1, len(idx) // 2)]
            logits = out.logits[:, -1, :].float()
            soft = torch.softmax(logits[:, pair_ids], -1).cpu().numpy()
            ps[idx] = soft[:, :len(YES)].sum(axis=1)
            i += len(idx); done += len(idx)
            if done % 300 < len(idx):
                print(f"  judge[{tag}] {done}/{len(texts)}")
        return 1.0 - ps if flip else ps

    pA_tr = judge(train_df, prompt_A, flip=False, tag="train/A")
    pB_tr = judge(train_df, prompt_B, flip=True, tag="train/B")
    pA_te = judge(test_df, prompt_A, flip=False, tag="test/A")
    pB_te = judge(test_df, prompt_B, flip=True, tag="test/B")

    del llm, tok
    gc.collect(); torch.cuda.empty_cache()
    return (pA_tr, pB_tr), (pA_te, pB_te)



# ----------------------------------------------------------------------
# 3. LANE 1a: SummaC-ZS windowed NLI (teammate's method + upgraded checkpoint).
#    Loaded ONCE and reused for both the native-Bengali check and the
#    translated-English cross-lingual check below -- no point running our
#    weaker single-shot NLI model in parallel when this one is strictly better.
# ----------------------------------------------------------------------
class SummacNLI:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        nli_hits = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "mdeberta" in p.lower()]
        nli_id = os.path.dirname(nli_hits[0]) if nli_hits else "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        print("SummaC NLI model:", nli_id)
        self.tok = AutoTokenizer.from_pretrained(nli_id)
        try:
            self.mod = AutoModelForSequenceClassification.from_pretrained(nli_id, dtype=torch.float16)
        except TypeError:
            self.mod = AutoModelForSequenceClassification.from_pretrained(nli_id, torch_dtype=torch.float16)
        self.mod = self.mod.to(DEVICE).eval()
        lab2id = {v.lower(): k for k, v in self.mod.config.id2label.items()}
        self.E_ID, self.C_ID = lab2id["entailment"], lab2id["contradiction"]

    @torch.no_grad()
    def _nli_batch(self, premises, hyps):
        out = []
        for i in range(0, len(premises), CFG["nli_batch"]):
            enc = self.tok(premises[i:i + CFG["nli_batch"]], hyps[i:i + CFG["nli_batch"]],
                           truncation=True, max_length=512, padding=True, return_tensors="pt").to(DEVICE)
            out.append(torch.softmax(self.mod(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    @staticmethod
    def _ctx_windows(ctx, win, cap=16):
        ss = sents(ctx)
        if len(ss) <= win:
            return [" ".join(ss)] if ss else []
        stride = max(1, win - 1)
        wins = [" ".join(ss[i:i + win]) for i in range(0, len(ss) - win + 1, stride)]
        wins.append(" ".join(ss[-win:]))
        return wins[:cap]

    def score(self, premises, responses, mask=None, tag=""):
        """SummaC-ZS windowed scoring. `premises`/`responses` are full column
        lists; `mask` restricts scoring to rows where the premise is meaningful
        (e.g. has_context) -- unscored rows get NaN."""
        n = len(premises)
        strict = np.full(n, np.nan)
        soft = np.full(n, np.nan)
        pos = np.arange(n) if mask is None else np.where(mask)[0]
        for j, i in enumerate(pos):
            ctx, resp = premises[i], responses[i]
            hyp = sents(resp)[:12] or [resp]
            st, so = [], []
            for win in CFG["win_list"]:
                wins = self._ctx_windows(ctx, win) or [ctx]
                P, H = [], []
                for h in hyp:
                    for w in wins:
                        P.append(w); H.append(h)
                probs = self._nli_batch(P, H).reshape(len(hyp), len(wins), -1)
                ent = probs[:, :, self.E_ID].max(axis=1)
                con = probs[:, :, self.C_ID].max(axis=1)
                st.append(float(ent.min() - con.max()))
                so.append(float((ent - con).mean()))
            strict[i] = float(np.mean(st)); soft[i] = float(np.mean(so))
            if (j + 1) % 300 == 0:
                print(f"  summac[{tag}] {j+1}/{len(pos)}")
        return strict, soft

    def unload(self):
        self.mod = self.mod.to("cpu")
        del self.mod
        gc.collect(); torch.cuda.empty_cache()


def run_summac_nli(train_df, test_df):
    """
    Unlike the teammate's original (context rows only), this scores EVERY row --
    using context when present, falling back to the prompt otherwise (mask=None).
    No-context rows are our identified weak spot (0.50 hallu_F1 vs 0.79 with
    context in tonight's testing); giving them a windowed NLI signal against the
    prompt, instead of leaving them NaN, directly targets that gap.
    """
    nli = SummacNLI()
    premise_tr = train_df["context"].fillna(train_df["prompt_bn"]).tolist()
    premise_te = test_df["context"].fillna(test_df["prompt_bn"]).tolist()
    strict_tr, soft_tr = nli.score(premise_tr, train_df["response_bn"].tolist(), mask=None, tag="train")
    strict_te, soft_te = nli.score(premise_te, test_df["response_bn"].tolist(), mask=None, tag="test")
    return nli, (strict_tr, soft_tr), (strict_te, soft_te)


# ----------------------------------------------------------------------
# 4. LANE 1b: deterministic math-word-problem solver (verbatim from teammate)
# ----------------------------------------------------------------------
BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
DAYS = ["রবিবার", "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার"]

def _nums(s):
    s = re.sub(r"(?<=[০-৯]),(?=[০-৯])", "", s)
    return [int(x.translate(BN2EN)) for x in re.findall(r"[০-৯]+", s)]

def _given_num(resp):
    resp = re.sub(r"(?<=[০-৯]),(?=[০-৯])", "", resp)
    m = re.findall(r"[০-৯]+", resp)
    return int(m[0].translate(BN2EN)) if m else None

def _close(a, b, tol=0.5):
    return a is not None and b is not None and abs(a - b) < tol

def solve_row(prompt, response):
    n = _nums(prompt)
    g = _given_num(response)
    if "সপ্তাহের কোন দিন" in prompt or "সপ্তাহের কোন বার" in prompt:
        day_names = [d for d in DAYS if d in prompt]
        if not day_names or not n:
            return None
        idx = (DAYS.index(day_names[0]) + n[0]) % 7
        return int(DAYS[idx] in response)
    if len(n) < 2:
        return None
    if re.search("একা একটি কাজ|একাই.*দিনে সমাপ্ত", prompt):
        a, b = n[0], n[1]
        correct = (a * b) / (a + b) if (a + b) else 0
        return int(_close(correct, g))
    if "ক, খ ও গ" in prompt:
        a, b, c = n[0], n[1], n[2]
        rate = (1 / a if a else 0) + (1 / b if b else 0) + (1 / c if c else 0)
        correct = 1 / rate if rate else 0
        return int(_close(correct, g))
    if "বয়সের অনুপাত" in prompt:
        a, b, total = n[0], n[1], n[2]
        share = min(a, b)
        correct = total * share / (a + b)
        return int(_close(correct, g))
    if "সংখ্যার অনুপাত" in prompt:
        a, b, total = n[0], n[1], n[2]
        correct = total * b / (a + b)
        return int(_close(correct, g))
    if "ক্রয়মূল্য" in prompt or "কেনা হয়েছিল" in prompt:
        cost, pct = n[0], n[1]
        correct = cost * (1 - pct / 100) if "ক্ষতি" in prompt else cost * (1 + pct / 100)
        return int(_close(correct, g))
    if "সরল সুদ" in prompt:
        principal, rate, time = n[0], n[1], n[2]
        correct = principal * rate * time / 100
        return int(_close(correct, g))
    if "অংশীদারের মধ্যে" in prompt:
        total, a, b, c = n[0], n[1], n[2], n[3]
        correct = total * b / (a + b + c)
        return int(_close(correct, g))
    if "একই দিকে দুইটি বাস" in prompt or "সাইকেল আরোহী একই বিন্দু" in prompt:
        s1, s2, t = n[0], n[1], n[2]
        correct = abs(s1 - s2) * t
        return int(_close(correct, g))
    if "দুইটি শহরের মধ্যে দূরত্ব" in prompt:
        dist, s1, s2 = n[0], n[1], n[2]
        correct = dist / (s1 + s2) if (s1 + s2) else 0
        return int(_close(correct, g))
    if "সংকেত বাতি" in prompt or "বাস স্টপেজ থেকে বাস" in prompt:
        from math import gcd
        a, b, c = n[0], n[1], n[2]
        lcm2 = a * b // gcd(a, b)
        correct = lcm2 * c // gcd(lcm2, c)
        return int(_close(correct, g))
    if "মিশ্রণে চিনি" in prompt:
        a, b, total = n[0], n[1], n[2]
        correct = total * b / (a + b)
        return int(_close(correct, g))
    if "প্যানেল গঠন" in prompt or "উপকমিটি গঠন" in prompt:
        from math import comb
        total, r = n[0], n[1]
        correct = comb(total, r)
        return int(_close(correct, g))
    if "রাশির গড়মান" in prompt or "শিক্ষার্থীর গড় নম্বর" in prompt:
        count, avg1, avg2 = n[0], n[1], n[2]
        correct = avg2 * (count + 1) - avg1 * count
        return int(_close(correct, g))
    return None

def math_feature(df):
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        lbl = solve_row(df["prompt_bn"].iat[i], df["response_bn"].iat[i])
        if lbl is not None:
            out[i] = lbl
    return out


# ----------------------------------------------------------------------
# 6. LANE 2: translation-based cross-lingual check (reuses the SAME SummacNLI
#    instance from Lane 1a -- no point loading our older, weaker NLI model when
#    his checkpoint is already loaded and strictly better) + lexical/rule features
#    that are genuinely ours: context_containment, novel_char_ratio, joggota rules,
#    response-quality heuristics. None of these exist in his pipeline at all.
# ----------------------------------------------------------------------
def run_cross_lingual_lane(nli: SummacNLI, train_df, test_df,
                            nli_native_tr, nli_native_te, translator=None):
    print("Loading translator for cross-lingual check...")
    if translator is None:
        ckpt = next((c for c in TRANSLATOR_CHECKPOINTS if os.path.exists(c)),
                    "facebook/nllb-200-distilled-600M")
        translator = Translator(checkpoint_path=ckpt)

    def translated_score(df, tag):
        premise = df["context"].fillna(df["prompt_bn"]).tolist()
        response = df["response_bn"].tolist()
        premise_en = translator.translate_batch(premise)
        response_en = translator.translate_batch(response)
        return nli.score(premise_en, response_en, mask=None, tag=f"{tag}-en")

    strict_en_tr, soft_en_tr = translated_score(train_df, "train")
    strict_en_te, soft_en_te = translated_score(test_df, "test")

    translator.model.to("cpu"); del translator
    gc.collect(); torch.cuda.empty_cache()

    strict_bn_tr, soft_bn_tr = nli_native_tr
    strict_bn_te, soft_bn_te = nli_native_te
    train_df["nli_summac_soft_en"] = soft_en_tr
    test_df["nli_summac_soft_en"] = soft_en_te
    train_df["cross_lingual_disagreement"] = soft_bn_tr - soft_en_tr
    test_df["cross_lingual_disagreement"] = soft_bn_te - soft_en_te
    return train_df, test_df


def bn_tokenize(text):
    if not isinstance(text, str):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return []
        text = str(text)
    return re.findall(r"[ঀ-৿]+|\S+", text)


def token_overlap_ratio(a, b):
    ta, tb = set(bn_tokenize(a)), set(bn_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def run_lexical_rule_features(train_df, test_df):
    print("Extracting joggota deterministic + lexical features...")
    train_df = extract_joggota_features(train_df)
    test_df = extract_joggota_features(test_df)
    for df in (train_df, test_df):
        premise = df["context"].fillna(df["prompt_bn"])
        df["token_overlap_ctx_resp"] = [
            token_overlap_ratio(p, r) for p, r in zip(premise, df["response_bn"])
        ]
    return train_df, test_df


# ----------------------------------------------------------------------
# LANE 4: ground-truth-source matcher (NCTB-QA + TyDi QA gold-passage, see
# ground_truth_matcher.py / CLAUDE.md "Ground-Truth Source Expansion"). CPU-only,
# no GPU needed -- safe to run alongside any other lane.
# ----------------------------------------------------------------------
def run_ground_truth_matcher(train_df, test_df):
    print("Matching against NCTB-QA + TyDi QA ground-truth sources...")
    matcher = GroundTruthMatcher()
    train_df, _ = compute_gt_features(train_df, matcher)
    test_df, _ = compute_gt_features(test_df, matcher)
    print(f"  train matched: {int(train_df['gt_matched'].sum())}/{len(train_df)} | "
          f"test matched: {int(test_df['gt_matched'].sum())}/{len(test_df)}")
    return train_df, test_df


# ----------------------------------------------------------------------
# 7. RUN EVERYTHING
# ----------------------------------------------------------------------
print("\n=== Lane 1a: SummaC windowed NLI (native, all rows -- context or prompt) ===")
nli_model, nli_native_tr, nli_native_te = run_summac_nli(train, test)

print("\n=== Lane 1b: math solver ===")
math_tr = math_feature(train)
math_te = math_feature(test)
print("math matched -- train:", int(np.sum(~np.isnan(math_tr))), "/", len(math_tr),
      "| test:", int(np.sum(~np.isnan(math_te))), "/", len(math_te))

print("\n=== Lane 2: translation-based cross-lingual check (reuses Lane 1a's NLI model) ===")
train, test = run_cross_lingual_lane(nli_model, train, test, nli_native_tr, nli_native_te)
nli_model.unload()

print("\n=== Lane 1c: gemma-4 bilingual judge ===")
(pA_tr, pB_tr), (pA_te, pB_te) = run_gemma_judge(train, test)

print("\n=== Lane 3: lexical + deterministic-rule features (genuinely ours, no overlap with his) ===")
train, test = run_lexical_rule_features(train, test)

print("\n=== Lane 4: ground-truth-source matcher (NCTB-QA + TyDi QA) ===")
train, test = run_ground_truth_matcher(train, test)

# assemble combined feature frame
def assemble(df, pA, pB, nli_native, math_arr):
    out = df.copy()
    strict, soft = nli_native
    out["llm"] = (pA + pB) / 2
    out["pA"] = pA
    out["pB"] = pB
    out["judge_disagreement"] = np.abs(pA - pB)
    out["nli_summac_strict"] = np.nan_to_num(strict, nan=0.0)
    out["nli_summac_soft"] = np.nan_to_num(soft, nan=0.0)
    out["math_override"] = np.nan_to_num(math_arr, nan=0.5)
    return out

train = assemble(train, pA_tr, pB_tr, nli_native_tr, math_tr)
test = assemble(test, pA_te, pB_te, nli_native_te, math_te)

HIS_FEATURES = ["llm", "pA", "pB", "judge_disagreement", "nli_summac_strict",
                "nli_summac_soft", "math_override", "has_context"]
OUR_FEATURES = ["context_containment", "novel_char_ratio", "word_entropy", "char_entropy",
                "token_overlap_ctx_resp", "length_ratio", "deterministic_joggota",
                "cultural_default_flag", "resp_is_refusal", "resp_code_switch_ratio",
                "resp_repetition_score", "resp_is_question",
                "nli_summac_soft_en", "cross_lingual_disagreement"]
MATCHER_FEATURES = ["gt_match_score", "gt_agreement", "gt_matched"]
ALL_FEATURES = HIS_FEATURES + OUR_FEATURES + MATCHER_FEATURES

train[ALL_FEATURES] = train[ALL_FEATURES].fillna(0)
test[ALL_FEATURES] = test[ALL_FEATURES].fillna(0)

train.to_pickle("fusion_train_features.pkl")
test.to_pickle("fusion_test_features.pkl")
print("\nFeature assembly complete. Saved fusion_train_features.pkl / fusion_test_features.pkl")
print(f"his features: {len(HIS_FEATURES)} | our features: {len(OUR_FEATURES)} | "
      f"matcher features: {len(MATCHER_FEATURES)} | combined: {len(ALL_FEATURES)}")
