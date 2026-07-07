# Native Bengali Hallucination Detection: The "Joggota" Framework

This document outlines a fully native, Bengali-first architecture for the Datathon 2.0 hallucination detection challenge. It abandons English-translation fallbacks (which fail on local context) in favor of analyzing the inherent structural and semantic truth of the Bengali language.

## Architecture Philosophy

We model every candidate response in a 3D linguistic space:
1. **আকাংক্ষা (Akangkha) & আসত্তি (Asotti)**: The "Form". Is the sentence syntactically coherent?
2. **যোগ্যতা (Joggota)**: The "Truth". Is the sentence factually, logically, and contextually valid?

Hallucinations are fundamentally failures of the **Joggota** axis.

## Open Questions

- **Model Selection:** For the local LLM Judge, our constraints (Kaggle offline) point towards an 8B model (LLaMA 3.1) or TigerLLM-9B. Does your team have the capacity to quantize Qwen 2.5 32B (via llama.cpp/GGUF) in a Kaggle kernel, or should we optimize for the 8B-9B tier?
- **Workflow Start:** Should we start implementing this directly within your `starter-notebook-datathon.ipynb` (to guarantee Phase 2 compliance from day one), or would you prefer I build it as standalone Python modules first?

## Proposed Architecture Modules

### 1. Data Ingestion & Pre-processing (The Foundation)
Cleanse the text, handle `[NULL]` contexts, and route the data.
- **Task Classifier:** RegEx-based routing to identify Idiom (বাগধারা), Math, Vocabulary, Grammar, or Factual questions.
- **Context Router:** Splits pipeline into RAG (Has Context) and Open-Domain (No Context).

### 2. The Form Engine (Akangkha & Asotti Scorers)
Captures structural deviations often found in hallucinations (verbosity, repetitive tokens).
- **Entropy Scorer:** Calculate character and word-level Shannon entropy. (Hallucinated responses tend to be abnormally verbose).
- **Perplexity Scorer:** Pass `response_bn` through a base Bengali LM (like BanglaBERT). High cross-entropy loss signals poor Asotti or factual dissonance.
- **Length Heuristics:** Basic character counts and length ratios between prompt and response.

### 3. The Truth Engine (Joggota Penalizers)
This is the core. We heavily penalize responses that deviate from the truth.

#### 3A. Contextual Joggota (For RAG / Has-Context Rows)
- **NLI (Natural Language Inference):** Use `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`. Compare `context` (Premise) against `response_bn` (Hypothesis). Output entails/contradicts probability.
- **Extrinsic Entity Penalty (SUST/BRACU NER):** Compute "Novel Character Ratio" and use local Bengali Named Entity Recognition to strictly penalize extrinsic hallucinations (e.g., inventing people/places not in context).

#### 3B. Factual Joggota (For Open-Domain / No-Context Rows)
- **Native LLM Judge:** Prompt an open-weight model (`TigerLLM-9B-IT`) to strictly evaluate the Bengali text. Force a single-token binary output (`yes`/`no` hallucinated).
- **Cross-Lingual Consistency:** Translate Bengali claim to English (`Helsinki-NLP/opus-mt-bn-en`), verify with English model. Extremely strong for general global facts (C0), but fails on local Bangladeshi facts (C1). Used as an additional feature, not the sole arbiter.
- **Chain-of-Thought (CoT) Math Joggota:** For math tasks, evaluate the reasoning steps (`arithmetic_slip`, `wrong_formula`), not just the final number.

#### 3C. Deterministic Lexical Joggota (The Rules Engine)
- **Idiomatic (বাগধারা) Joggota:** Flag literal interpretations of figurative language.
- **Bangla Academy Dictionary Fallback:** Bypass LLMs for `spelling` and `vocabulary` tasks. If the response violates strict lexical rules (e.g., verbosity in a spelling task, or missing from a local dictionary), auto-flag as hallucination. Highly efficient for offline Kaggle execution.

### 4. The Aggregation Meta-Classifier (The Final Arbiter)
Ensemble the dimensional scores into a final prediction.
- **XGBoost/LightGBM Classifier:** Train a gradient boosting tree on the 299 labeled samples (aligning with `tithy-4.ipynb`).
- **Features array:** `[NLI_probs, Cosine_sims, token_overlap, task_type, length_response, word_entropy, perplexity_score, novel_char_ratio, deterministic_joggota, llm_judge_binary]`
- **Calibration:** Set different decision thresholds based on the `task_type`.

## Submission Merge Strategy (Next Steps)
To prepare for our first official submission, we will merge our linguistic rules with the team's ML pipeline:
1. **Merge Codebases:** Extract the NLI, Embedding, and XGBoost logic from `tithy-4.ipynb` and combine it with the Form Engine and Deterministic Rules from `joggota_core.py` into a unified `submission_pipeline.py`.
2. **Feature Fusion:** The XGBoost model will train on the combined feature set, gaining semantic understanding (NLI) and strict structural verification (Entropy/Idioms).
3. **Execution:** Run the combined pipeline locally over `dataset samples.json` to train, predict on `test set.csv`, and output a ready-to-submit `submission.csv`.

## Verification & Execution Roadmap

1. **Phase 1: Form & Contextual Joggota (Fast Features)**
   - Implement the Entropy, Length, and Novel-char features.
   - Run the NLI model over the 132 context rows to establish a baseline F1 score.
2. **Phase 2: Factual Joggota (Heavy LLM)**
   - Set up the open-weight LLM pipeline locally/in Colab to score the 167 no-context rows.
3. **Phase 3: Meta-Model Integration**
   - Combine all extracted features into a DataFrame.
   - Train LightGBM with 5-fold cross-validation on the 299 training rows.
4. **Phase 4: Kaggle Phase-2 Compliance**
   - Package all models as Kaggle Offline Datasets.
   - Ensure the entire pipeline runs without internet access within the Kaggle time limit.

## User Review Required
> [!IMPORTANT]
> Please review this architecture. If you approve, let me know your preference on the **Open Questions**, and I will begin implementing the core Python code for the **Form Engine (Entropy)** and **Contextual Joggota (NLI)**.
