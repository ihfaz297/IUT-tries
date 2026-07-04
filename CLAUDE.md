# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**অলীকবচন** — Bengali LLM Hallucination Detection Challenge (Datathon 2.0, IUT / IPD / Brain Lab).

Binary classification: given a Bengali prompt and a candidate response (plus an optional context passage), predict whether the response is **faithful (label=1)** or **hallucinated (label=0)**. Evaluation metric is **macro F1** (binary F1 on the hallucinated class).

Two-phase competition:
- **Phase 1**: Submit prediction CSVs to Kaggle. Public + private leaderboard scoring.
- **Phase 2**: Top 30 teams submit a runnable solution package. Runs **offline in a Kaggle kernel — open-weight models only, no paid APIs**. Build Phase 2-compliant from day one.

## Data Files

| File | Description |
|---|---|
| `dataset samples.json` | 299 labeled examples — the only supervised signal available |
| `test set.csv` | 2,516 rows, no labels — submit predictions for these |
| `sample submission.csv` | Format: `id,label` (integers 1–2516, label 0 or 1) |
| `question_in_hand.txt` | Competition overview |
| `some_notes.txt` | Organizer notes — contains explicit winning approach hints |

### Schema

`dataset samples.json` — array of objects:
```
{
  "context":     "<Bengali passage>" | "[NULL]" | null,
  "prompt_bn":   "<Bengali question>",
  "response_bn": "<Bengali response>",  // sometimes stored as int — cast to str
  "label":       0 | 1
}
```

`test set.csv` — columns: `id, context, prompt_bn, response_bn` (no label).

## EDA Findings (299 labeled samples)

| Split | Rows | Hallucination rate |
|---|---|---|
| Has context | 132 (44%) | 36.4% |
| No context `[NULL]` | 167 (56%) | **52.7%** |

Null-context rows are harder and more hallucinated — the open-domain branch needs more signal than the RAG branch.

**Task type distribution (train → test):**
- Factual: 78.6% → 78.4% (stable, backbone of F1)
- Vocabulary: 10.4% → 11.8%
- Math: 9.0% → 5.8%
- Grammar: 1.0% → **3.3%** (3× higher in test — only 3 training examples, 0% hall. rate; don't over-trust)
- Translation / Spelling: <1%

**Hallucination rates by type:** Spelling 100% · Vocabulary 54.8% · Factual 44.7% · Math 44.4% · Grammar 0%

**Response length signal:** Hallucinated responses average 22 chars vs 13 for faithful — free meta-feature for LightGBM.

**Numeric regex (context rows only):** Precision 93%, Recall 29%, F1 44%. High-precision early flag, not a backbone signal.

## Source Paper

**BanglaHalluEval** (`README (1).md`) is the research benchmark this competition is built on. Reading it is high-leverage — it documents exact hallucination patterns, working judge prompts, and which open-weight models were tested.

### Hallucination Pattern Taxonomy (from BanglaHalluEval)

| Pattern | Where it appears | What it means |
|---|---|---|
| `factualness` | QA, Codemix | Response states a wrong fact |
| `comprehension` | QA, Codemix | Response misreads or ignores the context |
| `intrinsic` | Summarization | Response contradicts the source passage |
| `extrinsic` | Summarization | Response adds information not in the source |
| `arithmetic_slip` | Math reasoning | Correct formula, wrong calculation |
| `wrong_formula` | Math reasoning | Wrong formula applied |
| `wrong_unit` | Math reasoning | Correct calculation, wrong unit |

**Critical insight for math:** The paper evaluates the *entire reasoning chain*, not just the final answer. Errors are introduced at a specific `error_step`. A math response can reach the correct final answer via a flawed chain — or have a correct chain with only a unit error at the end. Check the chain, not just the number.

**Critical insight for context rows:** Intrinsic hallucinations (contradicting the source) and extrinsic hallucinations (adding facts) require different detectors. NER/regex catches extrinsic. NLI entailment is needed for intrinsic.

## Pipeline Design

Two branches based on `context == "[NULL]"`:

### Branch A — Has Context (RAG Faithfulness)
1. **Numeric/date regex mismatch** — numbers in response not present in context → near-certain extrinsic hallucination (Precision 93%, Recall 29% on sample split)
2. **NLI / semantic entailment** — `paraphrase-multilingual-MiniLM-L12-v2` cosine similarity between context and response; catches intrinsic contradictions
3. **Named entity overlap** — NER on both; entities in response absent from context signal extrinsic fabrication
4. **LLM judge (open-weight)** — use the prompt template below with TigerLLM or Qwen 2.5

### Branch B — No Context (Open-Domain)
1. **LLM judge (open-weight)** — zero-shot judge prompt with TigerLLM or Qwen 2.5; universal signal that works across all task types including C1 Bengali-specific facts
2. **Cross-lingual consistency** — translate Bengali claim to English (`Helsinki-NLP/opus-mt-bn-en`), verify with open-weight model. Strong for **C0 global facts only** — fails silently for C1 (Bangladesh-specific) because the English model has the same knowledge gap and produces no disagreement
3. **Math reasoning chain check** — for math problems, evaluate the full step-by-step chain, not just the final answer; target `arithmetic_slip`, `wrong_formula`, `wrong_unit`
4. **Vocabulary verification** — BanglaBERT or dictionary lookup for phrase/word meaning questions

> Cross-lingual is described as "strongest single signal" by organizers but they also explicitly warn: **do not translate-then-verify alone**. It is strong for C0 and useless for C1. The LLM judge covers both. Use cross-lingual as a *second independent signal* that boosts confidence on C0 rows, not as the primary detector.

### Meta-layer
- Assemble features from both branches into a **LightGBM classifier**
- Tune decision threshold **per task type** — not a single global threshold
- Baseline meta-features: `len(str(response_bn))` (+69% longer for hallucinated), task type one-hot, context present flag

### Entropy & perplexity features (free signals, add to LightGBM)

Confirmed signal on 299 samples (hallucinated vs faithful):

| Feature | Hall. | Faith. | Notes |
|---|---|---|---|
| Word entropy of response | 1.417 | 0.866 | Biggest gap; verbose = hallucinated |
| Char entropy of response | 3.228 | 2.867 | Partially redundant with length |
| Novel char ratio (context rows) | 0.040 | 0.014 | Chars in response not in context — extrinsic hallucination signal, independent of length |

```python
import math
from collections import Counter

def shannon_entropy(text):
    text = str(text)
    counts = Counter(text)
    total = len(text)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

def word_entropy(text):
    words = str(text).split()
    if len(words) < 2: return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

def novel_char_ratio(context, response):
    ctx_chars  = set(str(context))
    resp_chars = set(str(response))
    return len(resp_chars - ctx_chars) / max(len(resp_chars), 1)
```

**Perplexity (stronger, needs a model):** The LM-based version of entropy. Feed `response_bn` through TigerLLM or BanglaBERT and compute token-level cross-entropy loss. A Bengali LM assigns higher perplexity to responses that make unusual factual claims. Stronger than raw entropy but requires GPU inference. Implement after the LLM-judge signals are working.

## Judge Prompt Templates (from BanglaHalluEval — use directly)

All judge calls: `temperature=0`, max 8–32 tokens (32 for JSON tasks).

**QA / Factual (single-token):**
```
You are an evaluator.
Decide whether the provided model answer is hallucinated relative to the question.
Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.
Interpretation: 'yes' means the answer contains information not supported by the question/context
or is factually incorrect (hallucinated).
Provide the answer in English only: yes or no.

Question: {prompt_bn}
Model answer: {response_bn}

Answer now:
```

**Context-grounded / Summarization (single-token):**
```
You are an evaluator.
Decide whether the provided model answer is hallucinated relative to the document.
Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.
Interpretation: 'yes' means the answer contains information not supported by the document
or contradicts it (hallucinated).
Provide the answer in English only: yes or no.

Document: {context}
Summary: {response_bn}

Answer now:
```

**Math reasoning (JSON — use for TigerLLM and CoT):**
```
You are an evaluator for Bengali mathematical reasoning tasks.
Decide whether the provided reasoning chain is hallucinated relative to the question.
Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.
Interpretation: 'yes' means the chain or answer contains incorrect steps or unsupported facts.
Provide the answer in English only: yes or no.

Question: {prompt_bn}
Reasoning Chain: {response_bn}

Answer now:
```

**CoT variant (higher accuracy, more tokens):**
```
You are an evaluator checking whether a model answer is hallucinated.

Question: {prompt_bn}
Model Answer: {response_bn}

Analyze step by step:
Step 1: What factual claims does the answer make?
Step 2: Are these claims supported by or inferable from the question context?
Step 3: Based on steps 1-2, is the answer hallucinated?

Final answer (write only this word on the last line): yes or no
(yes = answer is hallucinated, no = answer is not hallucinated)
```

## Key Constraints & Organizer Hints (from `some_notes.txt`)

- **No large training set by design** — generalization across models is the goal, not overfitting to 299 examples
- **Cached frontier outputs** — GPT-4o, Claude, Gemini responses on the sample split are pre-released on Kaggle; use disagreement between these and the candidate response as features, no API budget needed
- **Cultural-default detection** — C1 hallucinations follow predictable patterns: wrong answer is the globally-dominant default (Nazrul → Tagore, Yunus Peace Prize → Economics). Explicit detectors for these win ground.
- **Do not translate-then-verify alone** — cross-lingual check is the strongest signal but not sufficient by itself
- **Per-band calibration** — C0 (global facts) / C1 (Bangladesh-specific) / C2 (contested/recent) bands are not disclosed, but calibrating per task type approximates this

## Task Type Classifier

Used throughout the pipeline to route rows and calibrate thresholds:

```python
import re

def classify_task(prompt: str) -> str:
    p = prompt.lower()
    if re.search(r'অর্থ|ভাবার্থ|শাব্দিক|সমার্থক|বিপরীত|প্রতিশব্দ', p):
        return "vocabulary"
    if re.search(r'বানান|শুদ্ধ বানান', p):
        return "spelling"
    if re.search(r'সম্ভাবনা|যোগ|বিয়োগ|গুণ|ভাগ|সংখ্যা|সমীকরণ|ক্ষেত্রফল|পরিসীমা|লসাগু|গসাগু', p):
        return "math"
    if re.search(r'অনুবাদ|ইংরেজি|translate', p):
        return "translation"
    if re.search(r'সমাস|ব্যাকরণ|কারক|বিভক্তি|ধাতু|প্রত্যয়|উপসর্গ', p):
        return "grammar"
    return "factual"
```

## Starter Notebook (`starter-notebook-datathon.ipynb`)

The official starter notebook establishes the floor. **Do not rebuild from scratch — extend it.**

### Confirmed results (5-fold CV on 299 samples)

| Approach | Scope | Macro-F1 | Notes |
|---|---|---|---|
| TF-IDF char 2-4gram + LogReg | All 299 rows | 0.5456 | Near-random; this is the submission floor |
| mDeBERTa NLI @ 0.5 threshold | Context rows only (130) | **0.6941** | Strong signal, confirmed working |

### Key patterns from the starter (reuse these)

```python
# Normalize context — handles [NULL], nan, empty, None
NO_CONTEXT_VALUES = {"", "nan", "NaN", "[NULL]", None}

def clean_context(value):
    if pd.isna(value) or str(value).strip() in NO_CONTEXT_VALUES:
        return ""
    return str(value).strip()

df["context"] = df["context"].apply(clean_context)
df["has_context"] = df["context"].str.len() > 0

# Always cast response_bn to str — some rows load as int from JSON
for col in ["prompt_bn", "response_bn"]:
    df[col] = df[col].astype(str)
```

### Confirmed NLI model (working in starter)

```python
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# zero-shot-classification pipeline, hypothesis_template="{}", multi_label=True
# Threshold at 0.5 → Macro-F1 0.69 on context rows
```

### The gap to close

The starter has **zero signal for null-context rows (169/299 = 56.5% of data, 52.7% hallucinated)**. That is the entire problem. Everything we build targets this gap — cross-lingual consistency, math solving, vocabulary verification.

## Models (all open-weight, Phase 2 compliant)

Tested and confirmed working in BanglaHalluEval — prioritize these:

| Model | Size | Best for | Notes |
|---|---|---|---|
| TigerLLM-9B-IT | 9B | Bengali-native judge | Use JSON output for math; HuggingFace local |
| Qwen 2.5 32B | 32B | General judge | Strongest open-weight judge in paper |
| Gemma2 27B | 27B | Summarization judge | Tested specifically on summ task |
| Mistral-Nemo | ~12B | Lightweight judge | Good CoT performance |
| DeepSeek-R1 14B | 14B | Reasoning/math | Best for chain-of-thought math |
| LLaMA 3.1 8B | 8B | Codemix, fast fallback | Lightest viable judge |
| `paraphrase-multilingual-MiniLM-L12-v2` | small | Sentence similarity | Lightweight, runs on CPU |
| `Helsinki-NLP/opus-mt-bn-en` | small | Bn→En translation | Cross-lingual consistency |
| BanglaBERT | ~110M | NLI / cross-encoder | Bengali-native BERT |
| LightGBM | — | Meta-classifier | Ensembles all features |

Kaggle GPU (P100/T4) can run up to ~13B models comfortably; 32B requires quantization (4-bit GGUF via llama.cpp) or is too slow for inference on 2,516 rows.
