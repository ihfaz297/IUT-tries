# Native Bengali Hallucination Detection: The Hybrid Cross-Lingual & LLM-Judge Framework

This document outlines a revised, leaderboard-optimized architecture for the Datathon 2.0 challenge. We are abandoning both the overfitted BanglaBERT approach and the pure rule-based "Joggota" heuristics. Instead, we are pivoting to a strategy that directly addresses the organizers' hints in `some_notes.txt`: **Cross-Lingual Consistency**, **Cultural-Default Detection**, and **Efficient Open-Weight LLMs**.

## Why the Previous Approaches Failed
1. **TigerLLM-9B**: OOM on Kaggle T4 due to model size and VRAM fragmentation.
2. **BanglaBERT**: Catastrophic overfitting to simple BenHalluEval "gibberish" artifacts; completely failed on complex reasoning and cultural-defaults (CV 0.823 → LB 0.602).
3. **Pure Heuristics**: Cannot capture complex factual errors (e.g., "Sarat Chandra" vs "Bankim Chandra") without external knowledge.

---

## The New Strategy: The "Dual-Brain" Efficient Pipeline

We will build a pipeline that leverages a **Small Multilingual LLM (1.5B–3B parameters)** that safely fits in a Kaggle T4's 16GB VRAM alongside our `mDeBERTa-v3` NLI model. 

### 1. The Light-Weight LLM Judge (Replacing TigerLLM-9B)
Instead of a 9B model, we will use **`Qwen2.5-1.5B-Instruct`** or **`Gemma-2-2B-it`**. 
* **Why?** These models require only ~3GB to 5GB VRAM at FP16. They are extraordinarily capable for their size, possess strong multilingual (Bengali) knowledge, and eliminate the OOM issues that killed the TigerLLM run.
* **Role**: For `no-context` rows, this model will act as a Zero-Shot factual judge. We will prompt it to directly evaluate if a candidate response is a hallucination.

### 2. Cross-Lingual Consistency (The "Silver Bullet")
As noted by the organizers, the strongest signal in this benchmark is the phenomenon where a model is correct in English but hallucinates in Bengali.
* **Mechanism**: 
  1. We translate the Bengali prompt into English using our small LLM or a fast NLLB model.
  2. We query the LLM for the answer in English (where its factual grounding is strongest).
  3. We compare the semantic similarity of the English answer to the Bengali candidate response (via LaBSE).
  4. If the English answer contradicts the candidate response, we heavily penalize the candidate.

### 3. Cultural Default Detector (C1 Band Focus)
The competition heavily features C1 "Cultural Default" questions (e.g., asking about a Bangladeshi inventor, but the response gives the Western default; or Nazrul vs Tagore).
* **Mechanism**: We will build an explicit dictionary of known "Global Defaults vs. Bangladesh Truths" (e.g., Tagore, Western Labs, Economics for Yunus). We will also prompt our LLM Judge to flag responses that seem to rely on global/Western defaults instead of Bangladeshi facts.

### 4. Enhanced Mini-RAG (For Context & Offline Corpus)
We will upgrade the `joggota_core.py` TF-IDF retriever. 
* **Direct Context Check**: For rows where context is provided, we run `mDeBERTa-v3` NLI(context, response) strictly. 
* **Corpus Retrieval**: For no-context rows, we retrieve the top paragraph from `offline_corpus.json`. If the similarity is above a high confidence threshold, we treat it as context and run NLI. 

### 5. Shallow XGBoost Fusion
All features (NLI Entailment, LaBSE Similarity, Cross-Lingual Agreement, LLM Judge Probability, Cultural Default Flag, and Heuristics) are fed into a conservative XGBoost classifier (`max_depth=4`, `n_estimators=400`) trained on the 299 labeled samples to calibrate the final probability threshold.

---

## Proposed Pipeline Execution

| Phase | Component | Action / Feature Generated |
|-------|-----------|----------------------------|
| **1** | **RAG & NLI Extraction** | Run `mDeBERTa-v3` and `LaBSE` on Context vs Response. For no-context rows, attempt to retrieve from `offline_corpus.json`. Flush models from VRAM. |
| **2** | **Small LLM Judge** | Load `Qwen2.5-3B-Instruct` (or 1.5B) in FP16. Generate the true English/Bengali answer for no-context rows. Compute `llm_judge_score`. Flush model. |
| **3** | **Heuristics & Defaults**| Run `joggota_core.py` to extract `novel_char_ratio`, `length_ratio`, and explicitly flag C1 `cultural_defaults`. |
| **4** | **Meta-Classifier** | Feed all generated features into XGBoost to predict the final hallucination label. |

---

## User Review Required
> [!IMPORTANT]
> **Model Selection for Phase 2**
> We propose using `Qwen/Qwen2.5-1.5B-Instruct` (or `3B-Instruct`) instead of BanglaBERT/TigerLLM. It has exceptional Bengali capabilities and fits well within the Kaggle T4 limit. Do you agree with testing Qwen2.5 for this LLM Judge phase?

## Open Questions
> [!WARNING]
> Do you have access to downloading `Qwen2.5-1.5B-Instruct` or `Gemma-2-2B-it` via Kaggle offline models for Phase 2, or should we prepare a script to download it into the pipeline for the Phase 1 submission? (Assuming we use internet in Phase 1 to download, then save to a Kaggle dataset for Phase 2).

## Verification Plan
1. Implement the Small LLM Judge in `submission_pipeline.py`.
2. Refactor NLI to properly separate context vs. no-context rows.
3. Run `diagnose.py` on the 299 sample dataset using this new pipeline to ensure the CV F1 score improves reliably (>0.75) without the BenHalluEval overfitting trick.
