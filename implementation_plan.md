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
- **Task Classifier:** RegEx-based routing to identify Math, Vocabulary, Grammar, or Factual questions.
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
- **Extrinsic Entity Penalty:** Compute "Novel Character Ratio". Punish responses that introduce entities (numbers, names) not present in the context.

#### 3B. Factual Joggota (For Open-Domain / No-Context Rows)
- **Native LLM Judge:** Prompt an open-weight model (`TigerLLM-9B-IT`) to strictly evaluate the Bengali text. Force a single-token binary output (`yes`/`no` hallucinated).
- **Chain-of-Thought (CoT) Math Joggota:** For math tasks, evaluate the reasoning steps (`arithmetic_slip`, `wrong_formula`), not just the final number.

### 4. The Aggregation Meta-Classifier (The Final Arbiter)
Ensemble the dimensional scores into a final prediction.
- **LightGBM Classifier:** Train a gradient boosting tree on the 299 labeled samples.
- **Features array:** `[task_type, length_response, word_entropy, perplexity_score, nli_entailment_prob, nli_contradiction_prob, novel_char_ratio, llm_judge_binary]`
- **Calibration:** Set different decision thresholds based on the `task_type` (e.g., lower threshold for Grammar, higher for Factual).

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
