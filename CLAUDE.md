# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**অলীকবচন** — Bengali LLM Hallucination Detection Challenge (Datathon 2.0, IUT / IPD / Brain Lab).

Binary classification: given a Bengali prompt and a candidate response (plus an optional context passage), predict whether the response is **faithful (label=1)** or **hallucinated (label=0)**.

**Metric — confirmed, not assumed**: LEGAL_DOCS.txt Section 7 states "Primary metric: binary F1 on the HALLUCINATED class (label = 0)." The Overview tab separately says "macro-F1" — these disagree, and the org has an open Discussion thread about it, but Section 7's wording is the more authoritative "Scoring, Ties, and Disputes" section, and a teammate's independently-built notebook (`phase2-final.ipynb`) reached the same conclusion in its own comments. Treat hallucinated-class F1 as primary. **This was a live bug for most of the competition**: `sklearn.metrics.f1_score(y, pred)` defaults to `pos_label=1`, which is the *faithful* class here — every CV number logged before the fix (including the BanglaBERT 0.823 postmortem below) was silently measuring the wrong class. Fixed in `submission_pipeline.py`/`fast_cv.py` by explicitly passing `pos_label=0`.

Two-phase competition:
- **Phase 1**: Submit prediction CSVs to Kaggle. Public + private leaderboard scoring.
- **Phase 2**: Top 30 teams submit a runnable solution package. Runs **offline in a Kaggle kernel — open-weight models only, no paid APIs**. Build Phase 2-compliant from day one.

## ⚡ Read this first — the lessons that actually moved the score

Everything in this repo took ~0.64 → **0.843**. Almost none of it came from modeling. Ranked by
what they were worth:

1. **Look it up before inferring it.** 3 lookup features (0.821) beat gemma-4 + SummaC-NLI +
   cross-lingual + math-solver combined (0.750). Months of model machinery lost to a TF-IDF
   index and a string compare. **Before building a model to predict X, check whether X is
   published somewhere.**
2. **When a source class stalls, ask what *shape* is missing — not how to get more.** Doubling
   the QA index changed coverage by literally zero rows (125 → 125). A different-shaped source
   (exam MCQs vs Wikipedia QA) took it to 210 and +0.119 real. **"We need data" is right far
   more often than it's actionable; the useful question is "which segment is dead, and what
   source is shaped like *it*?"**
3. **Small-sample OOF is a rough signal, not a ranking.** It picked the wrong *model* once
   (logreg 0.750 OOF → 0.692 real, while gnb 0.734 OOF → 0.717 real) and the wrong *threshold*
   once (0.78 tuned → real score identical to untuned). Track the per-classifier OOF→real gap
   (table in Leaderboard Status) and predict with it instead of trusting OOF raw.
4. **Segment your metric before optimizing it.** "0.717 overall" hid *0.787 on context rows vs
   0.500 on no-context* — the aggregate said "tune the model," the split said "half the test set
   has no signal at all." Every real gain after that came from the split, not the aggregate.
5. **A wrong number is worse than no number.** `f1_score()` defaults to `pos_label=1` = the
   *faithful* class here. Every CV number before that fix measured the wrong thing, including
   the BanglaBERT "0.823" that shipped a real 0.602.
6. **Gate-check "unavailable" claims.** BnMMLU is gated (401) — but an ungated 88K substitute
   (`hishab/bangla-mmlu`) did the same job. The blocker was real; the dead end wasn't.

## Current Pipeline Architecture

The pipeline is implemented in `submission_pipeline.py` and runs in **3 sequential phases** designed to never exceed a Kaggle T4 GPU's 16GB VRAM:

### Phase 1 — NLI + Embeddings + Cross-lingual (all rows)
- **mDeBERTa-v3** (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`): Produces entailment/contradiction probabilities between context (or prompt) and response — run twice, once in Bengali and once on NLLB-translated English text.
- **NLLB-200-distilled-600M** (`facebook/nllb-200-distilled-600M`): bn→en translation. Re-running NLI on the English translation and diffing against the Bengali verdict (`cross_lingual_disagreement`) implements the organizers' explicitly-stated strongest signal: "model correct in English, wrong in Bengali."
- **LaBSE** (`sentence-transformers/LaBSE`): Cosine similarity between context/prompt embedding and response embedding.
- **Joggota Engine** (`joggota_core.py`): Deterministic rule-based features (entropy, length ratio, novel char ratio, task classification, idiom detection, context containment, response-quality heuristics).
- After features are extracted, **all three models are deleted and GPU memory is flushed** (`gc.collect()` + `torch.cuda.empty_cache()`).

### Phase 2 — Small LLM Judge (no-context rows only)
- **NO TigerLLM**: Attempted 4-bit quantization on Kaggle T4 — OOM errors. The 9B model needs ~8GB at 4-bit and T4 has 16GB but with tokenizer overhead + VRAM fragmentation, it breaks. Dead end on current hardware.
- **NO BanglaBERT**: Fine-tuned on 2.5K BenHalluEval QA pairs + 299 competition samples. CV F1 showed 0.823 but actual leaderboard score dropped to 0.602 — learned a cheap shortcut (non-Bengali tokens in BenHalluEval's gibberish hallucinated samples) instead of real faithfulness discrimination. Made things worse.
- **Current**: `SmallLLMJudge` — Qwen2.5-1.5B-Instruct, logit-based scoring (P(token='1') vs P(token='0') at the final position, not fragile string-matched generation), batched. Scores only no-context rows. Falls back to `sim_premise_response` if the checkpoint can't load.

### Phase 3 — XGBoost Fusion
- **XGBoost** (`max_depth=3`, `n_estimators=300`): Shallow ensemble to prevent overfitting on 299-sample training set.
- Feature set (see `build_base_features` / `extract_joggota_features` for the authoritative list — do not trust a hardcoded count here, it drifts):
  - `nli_ctx_entail`, `nli_ctx_contra` — bn NLI scores
  - `nli_en_entail`, `nli_en_contra`, `cross_lingual_disagreement` — English-translated NLI + disagreement vs bn verdict
  - `sim_premise_response`, `xlingual_consistency` — LaBSE cosine similarity
  - `token_overlap_ctx_resp` — Lexical overlap
  - `has_context` — Binary flag
  - `word_entropy`, `char_entropy` — Response randomness
  - `novel_char_ratio` — Extrinsic hallucination signal
  - `length_ratio` — Response vs prompt length
  - `deterministic_joggota` — Rule-based verdict
  - `cultural_default_flag` — C1 band cultural default detection
  - `context_containment` — Response bigrams verbatim in context
  - `resp_is_refusal`, `resp_code_switch_ratio`, `resp_repetition_score`, `resp_is_question` — Response quality heuristics
  - `llm_judge_score` — Qwen2.5-1.5B logit-based faithfulness score (no-context rows; `sim_premise_response` elsewhere)

## Fusion Pipeline (`fusion_pipeline.py` + `fusion_evaluate.py` — the current, more advanced track)

A teammate (Alvee) independently built `phase2-final.ipynb`: gemma-4-12b-it bilingual (bn+en)
judge, SummaC-ZS windowed NLI via a stronger checkpoint
(`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`), a 15-template deterministic
math-word-problem solver, and a GaussianNB decision layer. It scored a **real, Kaggle-confirmed
0.777** — a large jump over our own 0.636. The fusion pipeline combines the two approaches
rather than picking one:

- **Replaces** our old single-shot NLI with his SummaC windowed NLI entirely (strictly better;
  no point running both). Extended to score *every* row, not just context rows (his original
  only covered context rows) — no-context rows use the prompt as a fallback premise, directly
  targeting our worst-performing segment (see "No-context rows" below).
- **Reuses** that same loaded NLI model (not a second one) to score NLLB-translated English
  text too, so our translation-based cross-lingual check rides on his stronger checkpoint
  instead of our older weaker one — one model loaded, not two.
- **Keeps near-verbatim**: his gemma-4-12b-it judge and math solver.
- **Adds, genuinely ours, no overlap with his pipeline**: `context_containment` and
  `novel_char_ratio` (our two highest-importance features in the real Kaggle run —
  see Leaderboard Status), the fixed deterministic joggota rules, response-quality heuristics.
- **Evaluation**: proper `StratifiedKFold` OOF comparison across (his features / ours /
  matcher / his+matcher / our+matcher / combined) × (GaussianNB / LogisticRegression /
  XGBoost) — picks whichever wins on OOF hallu-F1 rather than assuming fusion is
  automatically better. Threshold is the *average of each fold's own held-out best
  threshold*, not a single value grid-searched against the full OOF pool (see the
  overfitting lesson below — this is a direct mitigation for it).
- **Lane 4, added later (`ground_truth_matcher.py`)**: the ground-truth-source matcher —
  see "Ground-Truth Source Expansion" below for full detail. CPU-only, indexes NCTB-QA +
  TyDi QA and produces `gt_match_score`/`gt_agreement`/`gt_matched` as three soft features,
  kept in their own `MATCHER_FEATURES` list (not dumped into `OUR_FEATURES`) specifically so
  `fusion_evaluate.py` can isolate its effect from the binary/rare-firing rule features that
  are what caused GaussianNB to collapse earlier.

`fusion_converter.py` assembles `joggota_core.py` + `submission_pipeline.py` (model class defs
only — see the `__main__`-guard gotcha below) + `fusion_pipeline.py` + `fusion_evaluate.py` into
one self-contained `fusion_kaggle.ipynb`, split across cells so a gemma-4 smoke test (Part 2.7,
3 rows) can run before the expensive full 2,815-row lanes (Part 3) — catches a load/version
failure in under a minute instead of after burning through the NLI lane first.

### Bugs found and fixed while building this (don't re-discover these)
- `submission_pipeline.py` ends with `if __name__ == "__main__": train_and_predict()`. Inside a
  notebook cell `__name__` **is** `"__main__"` — inlining that file verbatim would wrongly
  auto-run the old standalone pipeline the moment the cell executes. `fusion_converter.py`
  strips this guard via regex before inlining.
- The smoke-test cell needs `train` and `run_gemma_judge` to already exist. Both are defined
  partway through `fusion_pipeline.py`'s "Lane 1c" section, which originally came *after* the
  smoke test in notebook order. Fixed by physically moving the gemma-4-judge lane earlier in
  `fusion_pipeline.py` (right after data loading), not by reordering notebook cells around a
  fragile text split.
- `token_overlap_ctx_resp` was listed in `OUR_FEATURES` but the code to actually compute it
  (`bn_tokenize`/`token_overlap_ratio`, originally only in the old `submission_pipeline.py`) was
  never ported into `fusion_pipeline.py`. Caused a `KeyError` — but only *after* all four
  expensive GPU lanes (NLI, math, cross-lingual, gemma-4 judge) had already finished on the full
  2,516-row test set. **Lesson**: cross-check every name in a feature list is actually assigned
  somewhere before trusting a long pipeline to reach the end — `grep` for `"feat"] =` per feature
  name, don't eyeball it.

### gemma-4 / transformers version gotcha
`gemma4_unified` (gemma-4's architecture) is not recognized by **any released PyPI version** of
`transformers` — confirmed empirically, not assumed: even the latest release (5.13.1) fails with
`KeyError: 'gemma4_unified'`. Needs an install from GitHub source
(`pip install git+https://github.com/huggingface/transformers.git`). This is almost certainly
what the teammate's "wheelhouse" dataset actually contained (a source build for offline/Phase-2
installs), not just "any transformers>=5.13" as his comment implied. `bootstrap_transformers()`
in `fusion_pipeline.py` now: tries a plain PyPI upgrade first (fine for Phase 1, internet on),
*verifies* `gemma4_unified` is actually in `CONFIG_MAPPING` (not just checks a version number —
confirmed a high version number alone is insufficient), and falls back to a source install if
not. **After any transformers reinstall, the Kaggle kernel must be restarted**, not just have
cells re-run — Python caches the already-imported module in memory; a fresh install on disk
doesn't retroactively swap out what's already loaded in a running session.

### Kaggle API auth gotcha
The `kaggle` CLI (2.2.3+) has moved to OAuth (`kaggle auth login`) and the classic static-key
`kaggle.json` method is **rejected server-side with 401**, confirmed by bypassing the CLI and
hitting the Python `KaggleApi` class directly — not a client tooling issue, Kaggle's backend no
longer accepts it. `kaggle auth login` (run via Bash on the user's own machine) opens their real
local browser and completes automatically if already logged into kaggle.com there. **Trap**: a
stale legacy `kaggle.json` sitting in `~/.kaggle/` will silently shadow working OAuth credentials
stored separately (`~/.kaggle/credentials.json`) and produce the exact same generic
"Authentication required" error either way — move/delete the legacy file if OAuth login reports
success but calls still fail. Also: `kernels status`/logs API endpoints only work for
**committed** ("Save & Run All") runs, not live interactive cell-by-cell sessions — there's no
session object to query for the interactive mode. `kernels pull` *does* work for interactive
sessions, but only returns saved source code, never outputs/errors.

## File Map

| File | Description |
|---|---|
| `submission_pipeline.py` | Original inference pipeline — 3-phase architecture. Superseded in ambition by the fusion pipeline but its model classes (`Translator`, `NLIScorer`, `Embedder`, `SmallLLMJudge`) are still reused/inlined by it |
| `joggota_core.py` | Deterministic rules engine + Form Engine (no retrieval anymore — see below) |
| `fusion_pipeline.py` | **Current best-track pipeline** — combines teammate's gemma-4/SummaC-NLI/math-solver with our cross-lingual/lexical features, plus Lane 4 (ground-truth matcher). See "Fusion Pipeline" section above |
| `fusion_evaluate.py` | Honest OOF CV comparison (his / ours / matcher / his+matcher / our+matcher / combined × 3 model types) + final submission generation, reused from pickles `fusion_pipeline.py` writes. Writes 3 CSVs: `submission_fusion.csv` (OOF-picked winner), `submission_his_gnb_baseline.csv` (fixed his-alone+gnb reference), `submission_his_matcher_gnb.csv` (fixed his+matcher+gnb — new) |
| `fusion_converter.py` | Assembles `joggota_core.py` + `ground_truth_matcher.py` + `submission_pipeline.py` + `fusion_pipeline.py` + `fusion_evaluate.py` → `fusion_kaggle.ipynb` |
| `evaluate_source.py` | ⭐ **The method, not just the result.** Given any candidate `(question, answer)` DataFrame, reports incremental coverage on no-context rows + corr, and gives an ADD/SKIP verdict. Encodes the "shape beats size" lesson so the next source hunt is systematic instead of guesswork. Already used to kill `titulm-bangla-mmlu` (+0 rows) in 30s. **Run this before adding anything.** |
| `ground_truth_matcher.py` | ⭐ **The single highest-value file in the repo — it produced 0.843.** Indexes **5 ungated sources (105,262 QA pairs)**, all auto-downloading from HuggingFace if missing: NCTB-QA, TyDi QA, IndicQA, BanglaRQA, **bangla-mmlu**. Char-ngram TF-IDF, near-exact question match (cosine ≥0.75), numeric-aware answer agreement. Emits `gt_match_score`/`gt_agreement`/`gt_matched`. Has a `__main__` self-test (stripped by `fusion_converter.py`) that prints coverage/corr against the 299 train rows |
| `nctb_qa_train.parquet` | Cache: NCTB-QA Bengali+English train (12,203 rows; 5,812 Bengali) from `ShihabReza/nctb-qa`. Auto-redownloads if missing |
| `tydiqa_goldp_bengali.parquet` | Cache: TyDi QA Bengali gold-passage (2,503 = 2,390 train + 113 val), filtered from `google-research-datasets/tydiqa` `secondary_task`. Prefers `curious_insight/`'s CSV if present, else this cache, else re-downloads+re-filters |
| `bangla_mmlu.parquet` | ⭐ Cache: **`hishab/bangla-mmlu`, 87,694 Bengali exam MCQs** (validation+test splits), answer-letter already resolved to option text. **The free substitute for gated BnMMLU, and the source that unlocked the no-context rows.** Auto-redownloads if missing |
| `indicqa_bn.json` | Cache: IndicQA Bengali (1,263 usable QAs) from `ai4bharat/IndicQA`, CC-BY-4.0. Measured to add **zero** coverage on top of NCTB+TyDi — kept only because test distribution may differ |
| `banglarqa_train.json` | Cache: BanglaRQA (7,990 usable QAs) from `sartajekram/BanglaRQA`. Loader drops unanswerable rows and yes/no "confirmation" questions (a "হ্যাঁ"/"না" answer can't discriminate faithful from hallucinated). Also added ~zero coverage — same caveat |
| `fusion_kaggle.ipynb` | Auto-generated — **the notebook actually being run on Kaggle right now** |
| `phase2-final.ipynb` | Teammate's (Alvee) original notebook — real Kaggle-confirmed **0.777**. Source of the gemma-4/SummaC-NLI/math-solver logic fusion_pipeline.py adapts |
| `hallucination-v1/v2/v3.ipynb` | Another teammate's ensemble notebook (fine-tuned mDeBERTa+BanglaBERT+XLM-R, 5-fold CV). ⚠️ v2's actual test predictions were 96.5% hallucinated vs train's 45.5% — a red flag for severe miscalibration on 299-row fine-tuning. Never submitted; not the track being pursued |
| `fast_cv.py` | Train-set-only (299 rows) CPU-friendly CV harness for fast local iteration, with feature ablation |
| `fast_cv_rerun.py` | Reuses `fast_cv_train_df.pkl` to rerun just the LLM judge + CV without redoing the slow translation/NLI extraction |
| `fast_cv_model_compare.py` | XGBoost vs GaussianNB vs LogisticRegression comparison — GaussianNB alone was *much worse* (0.23), LogReg roughly tied XGBoost. Model swap alone doesn't explain a competitor's higher score |
| `fast_cv_tfidf_nb.py` | TF-IDF + Naive Bayes on raw text (not engineered features) — also underperformed (0.28-0.59). Ruled out as an explanation too |
| `converter.py` | Converts `submission_pipeline.py` → `submission_kaggle.ipynb` (the older, single-approach notebook) |
| `submission_kaggle.ipynb` | Auto-generated from `submission_pipeline.py` — superseded by `fusion_kaggle.ipynb` |
| `train_banglabert.py` | ❌ Failed BanglaBERT fine-tuning script (do not use — makes scores worse) |
| `augment_from_benhallueval.py` | ❌ Failed data augmentation — extracts noisy training pairs from BenHalluEval |
| `benhallueval_training.json` | ❌ Noisy QA pairs — 5K hallucinated, 2.5K faithful (caused the 0.823→0.602 drop) |
| `curious_insight/` | ⚠️ The BanglaHalluEval benchmark repo (LLM-generated hallucinations, GPT-4.1-mini/Ollama judged) — a *different* project entirely, not competition data. Same trap as the BenHalluEval training failure below; not used, not committed to git |
| `dataset samples.json` | 299 labeled examples — the only supervised signal |
| `test set.csv` | 2,516 rows, no labels — submit predictions for these |
| `sample submission.csv` | Format: `id,label` |
| `question_in_hand.txt` | Competition overview |
| `some_notes.txt` | Organizer notes — winning approach hints |
| `some_catches.txt` | Phase 2 preview — source domains, methodology refs (does **not** contain the retrieval corpus itself — that's in a separate, newer Discussion post, still unfetched as of this writing) |
| `LEGAL_DOCS.txt` | Full competition rules — Section 7 has the authoritative metric definition |
| `freelance.txt` | Example of cultural stereotype hallucination (Humayun Ahmed vs Taslima Nasrin) |
| `check_noise.py` | Diagnostic: spot-checks BenHalluEval training data quality |
| `check_submission.py` | Diagnostic: compares submission distributions across runs |
| `diagnose.py` | Diagnostic: analyzes CV F1 vs actual score gap |
| `run_log.csv` | Previous run log (CV F1 0.646, TigerLLM fallback) |
| `run_log (1).csv` | Real Kaggle-run log, corrected-metric era — `cv_f1_mean` here is the actual hallu-F1 (0.6362 @ threshold 0.5) |

## Joggota Engine (`joggota_core.py`)

### Form Engine Features
- `word_entropy` / `char_entropy` — verbose hallucinations have higher entropy
- `novel_char_ratio` — characters in response not in context (extrinsic hallucination)
- `length_ratio` — hallucinated responses are ~69% longer
- `resp_len` — raw response length

### Task Classifier
Routes prompts into: `idiom`, `vocabulary`, `spelling`, `math`, `translation`, `grammar`, `factual`. 249/299 training rows fall into `factual` (no rule applies) — deterministic rules can only ever touch a small minority of rows; don't expect them to move the aggregate metric much even when correct.

**Bug fixed**: math classification used to `re.search()` short keywords ("ভাগ"=divide, "যোগ"=add) as raw substrings against the whole prompt — which false-matched inside unrelated words like "বিভাগ" (department) and "প্রতিযোগিতা" (competition), since Bengali suffixes attach directly to stems with no space. Misclassified wrestling-championship and civics questions as math. Fixed by checking whether a whitespace-split *token* **starts with** the keyword instead (`classify_task` in `joggota_core.py`) — real math terms take the keyword as a prefix (e.g. "সংখ্যার" starts with "সংখ্যা"), false matches buried mid-word don't.

### Deterministic Rules
- **Spelling**: Response >5 words = hallucination (should be a single corrected word)
- **Math**: No digits in response = hallucination. **Bug fixed**: this missed spelled-out Bengali number words ("সাতটি" = "seven") and MCQ letter answers ("গ" = option C), both valid non-hallucinated responses with no literal digit. Now also checks against a Bengali number-word list and single-letter MCQ options (`_BN_NUMBER_WORDS`, `_MCQ_OPTIONS` in `joggota_core.py`). Before the fix: fired on 8 rows, right on only 3. After: fires on 1 row, right on 1 — precision fixed, but reach is now so small it can't meaningfully move the aggregate score on its own.
- **Idioms**: Hardcoded dictionary of 9 common বাগধারা with literal vs figurative keyword checks. If LLM gives literal meaning instead of figurative → instant `0.0`. Doesn't appear at all in the 299-row training sample — untested against ground truth locally.

### Context Grounding & Response Quality
- `context_containment` — fraction of response's word bigrams that appear verbatim in the context (direct lift = strong faithfulness signal).
- `resp_is_refusal` — flags deflection phrases ("আমি জানি না", "দুঃখিত", etc).
- `resp_code_switch_ratio` — ratio of Latin-alphabet tokens in the response.
- `resp_repetition_score` — internal bigram repetition.
- `resp_is_question` — response deflects by ending in "?".

### ⚠️ Removed: Mini-RAG (Corpus Retriever)
The self-built TF-IDF retriever over a Wikipedia crawl (`build_corpus.py` → `offline_corpus.json`, ~441 paragraphs) was **deleted** — it was never the organizer-provided retrieval corpus the welcome note references ("pull evidence from the provided Bengali corpus"), just a stand-in, and its `corpus_match_score` feature was unvalidated.

**Status as of this writing**: organizer (Sushmit) confirmed in a Discussion reply that the real corpus *has* been shared, in a **new, separate Discussion post** (distinct from `some_catches.txt`'s Phase 2 preview note, which was double-checked and does not contain it) — "covers both phase-I and II." **That post has not been fetched/read yet.** He also confirmed the cached frontier-model outputs (GPT-4o/Claude/Gemini judgments, the "strongest single signal" per the welcome note) are **not released yet** — "will release... keep watching discussions." Finding and using the real corpus is still the highest-leverage unclaimed lever, especially for no-context rows (see Leaderboard Status) since it's the only thing that could give them actual external grounding instead of internal-consistency-only signal.

## Leaderboard Status

### 🏆 Current best: **0.843 real, Kaggle-confirmed** (`submission_fusion.csv` = his+matcher+logreg @ thr 0.52)

Beats teammate Alvee's 0.777. The entire jump came from **one thing**: the ground-truth matcher
reaching the no-context rows via an exam-bank source (`hishab/bangla-mmlu`). See "Ground-Truth
Source Expansion" for the full mechanism — it is the single most important section in this file.

**Real score progression (all Kaggle-confirmed):**
| submission | config | OOF hallu_F1 | real | note |
|---|---|---|---|---|
| pre-fusion | our XGB pipeline | — | 0.636 | cross-lingual era |
| `submission_fusion.csv` (v1) | his-alone + logreg | 0.7500 | 0.692 | OOF "winner", lost in reality |
| `submission_his_gnb_baseline.csv` | his-alone + gnb | 0.7336 | **0.717** | OOF runner-up, won in reality |
| `submission_fusion.csv` (v2) | combined + logreg, **2-source matcher** | 0.7708 | **0.724** | matcher's first real gain |
| `submission_fusion.csv` (v3) | his+matcher + logreg, **5-source matcher** | 0.8683 | **0.843** | ⬅ current best |

**OOF→real calibration, empirically tracked** (use this to predict, don't trust OOF raw):
- gnb: 0.7336 → 0.717 (**-1.7pp**, tight)
- logreg: 0.7500 → 0.692 (-5.8pp), 0.7708 → 0.724 (-4.7pp), 0.8683 → 0.843 (**-2.5pp**)
- logreg's OOF overshoots by ~2.5-5.8pp. The overshoot *shrank* as real signal replaced fitted
  noise — the 0.843 run had the smallest gap, consistent with the matcher adding genuine
  information rather than the model fitting the 299 rows harder.

- **Old notes below are superseded but kept — the lessons are still live:**
- **Two real, Kaggle-confirmed scores from the fusion session, and they invert the OOF ranking:**
  - `submission_his_gnb_baseline.csv` ("his features alone + gnb", OOF hallu_F1=0.7336, the OOF *runner-up*) → **real score 0.717**
  - `submission_fusion.csv` ("his features alone + logreg", OOF hallu_F1=0.7500, the OOF *winner*) → **real score 0.692**
  - **⚠️ OOF model-selection picked the wrong winner, not just the wrong threshold.** Earlier tonight the lesson was "threshold tuning overfits on 299 rows." This is the same failure mode one level up: choosing *which model* based on OOF hallu-F1 also isn't reliable at this sample size — the OOF-worse config (gnb, 0.7336) beat the OOF-better one (logreg, 0.7500) by 2.5 points in reality. Small-sample OOF comparisons should be treated as rough signal, not a trustworthy ranking, for model choice *and* threshold *and* feature-set decisions.
- **Teammate Alvee's real, Kaggle-confirmed score: 0.777** — but that's **v21**, not the v19 architecture `phase2-final.ipynb` actually contains. Precisely traced: v19's own markdown claims "validated Phase-1 LB 0.739, wcv 0.7777" — two different numbers. **"wcv 0.7777" is confirmed to be the in-sample metric, not real validation** — it's the exact same figure printed by cell `afd48b33`'s `in-sample GNB thr=0.180 macro-F1=0.7774` (fit-on-X, score-on-same-X). The real, externally-validated v19 number is **0.739**. `submission_his_gnb_baseline.csv` (our reproduction of v19's architecture: SummaC NLI + gemma-4 judge + math solver + GaussianNB) scored a real **0.717** — close to 0.739, the gap explained by real differences (we extended SummaC NLI to score every row via prompt-fallback, not just context rows; different threshold-tuning method). **0.777 = v19 + a BnMMLU exam-bank answer-key override** (110 high-confidence exact-match rows, 42 flips, 79 total deterministic flips vs v19) — so the override's isolated real contribution is ≈0.739→0.777, **roughly +0.038**, a concrete quantified lever the fusion pipeline does not have yet. See "Ground-Truth Source Expansion" below.
- **The OOF comparison genuinely tested fusion vs. no-fusion, and fusion did not win on OOF either way.** Real printed results (n=299, `fusion_evaluate.py`):
  ```
  his features alone + gnb        hallu_F1@avg_thr=0.7336
  his features alone + logreg     hallu_F1@avg_thr=0.7500   <- winner
  his features alone + xgb        hallu_F1@avg_thr=0.7305
  our features alone + gnb        hallu_F1@avg_thr=0.0699   <- catastrophic
  our features alone + logreg     hallu_F1@avg_thr=0.6787
  our features alone + xgb        hallu_F1@avg_thr=0.7378
  combined (fusion) + gnb         hallu_F1@avg_thr=0.1224   <- catastrophic
  combined (fusion) + logreg      hallu_F1@avg_thr=0.7443
  combined (fusion) + xgb         hallu_F1@avg_thr=0.7417
  ```
  Winner was **"his features alone + logreg"** — the combined feature set never beat his-alone. Both `submission_fusion.csv` and `submission_his_gnb_baseline.csv` ended up using his 8 features only (different classifiers: logreg vs gnb) — **despite the filename, `submission_fusion.csv` does not actually use our added features**, because the combined config didn't win. Our cross-lingual/lexical additions did not demonstrably help on this OOF measure (0.744-0.742 combined vs 0.750 his-alone — a gap small enough to be noise on 299 rows, but not evidence *for* fusion either; if the added features carried strong signal we'd expect combined to clearly beat his-alone, not trail it).
- **GaussianNB catastrophically fails the moment our features are added**: 0.070 on our features alone, 0.122 combined, vs 0.734 on his features alone. Confirms the earlier local finding (GNB is Gaussian-per-class; our features include binary/rare-firing rule flags that badly violate that assumption) at much larger scale. His 8 features (judge probabilities, NLI scores) are smooth and continuous — a good GNB fit. Not a fusion-concept problem, a GNB-specific one — logreg/xgb don't show this collapse.
- **Old scores, superseded**: 0.636 (pre-fusion cross-lingual pipeline), 0.602 (BanglaBERT experiment, see postmortem below). Neither is the current baseline.
- **No-context rows are the primary bottleneck, quantified**: local OOF testing showed **0.787 hallu-F1 on context rows vs 0.500 on no-context rows** — barely better than chance. Root cause: our two highest-importance features (`novel_char_ratio` at 0.12 importance, `context_containment` at 0.11) both structurally return 0 on no-context rows (nothing to compare the response against). This is *why* the fusion pipeline's SummaC lane was extended to score every row using the prompt as a fallback premise instead of leaving no-context rows blank.
- **Train/test distribution mismatch**: train is 43.5% context rows, test is 54.1% — an 10.6pp gap. Worth remembering when trusting CV numbers.
- **⚠️ OOF threshold tuning overfits on 299 rows — proven, not theoretical.** Grid-searching a single global threshold against pooled OOF predictions found 0.78 (hallu-F1 0.7139 in validation) — but the *actual* submitted leaderboard score was 0.636, exactly matching the untuned @0.5 baseline. The "improvement" was fitting noise in a small validation pool, not a real gain. `fusion_evaluate.py` mitigates this by averaging each fold's own independently-found best threshold instead of searching the pooled OOF set once — still imperfect with this little data, but structurally less prone to the same failure.
- **Real feature importances** (from the actual Kaggle run, not local CV): `novel_char_ratio` (0.120) and `context_containment` (0.112) — both cheap, lexical, non-model features — outweighed the entire cross-lingual NLI+translation stack combined (~0.122) and roughly tied the whole Qwen judge (0.067) contribution. The expensive multi-model machinery was real but underpowered relative to simple lexical heuristics; the fusion pipeline's bet is that gemma-4's much larger judge changes this balance.

## Ground-Truth Source Expansion — ✅ BUILT, KAGGLE-CONFIRMED, THIS IS WHAT GOT 0.843

### The one finding that matters: source *shape* beats source *size*, and the two shapes are disjoint

Measured on the 299 train rows, this is the whole game:

| source type | context rows matched | **no-context rows matched** |
|---|---|---|
| Wikipedia-derived QA (NCTB, TyDi, IndicQA, BanglaRQA) | **92.4%** | **1.8%** |
| Exam MCQ bank (`hishab/bangla-mmlu`) | 0.8% | **38.9%** |
| **All five combined** | **93.9%** | **51.5%** |

**They are near-perfectly complementary.** This is not a coincidence — it reveals how the
competition test set was built:
- **Context rows were lifted from public Bengali QA datasets.** The `context` field is the
  passage, `prompt_bn` is the question, *verbatim*. 112 train rows match at cosine **1.000**
  (exact string match), and **all 112 have context**. `gt_agreement` isn't inferring
  faithfulness — it's checking the response against the source dataset's own answer key.
- **No-context rows are exam-bank shaped** (general knowledge, literature, vocabulary), which
  is why Wikipedia QA can't touch them and an MCQ bank can.

**Consequence — the negative result that nearly stopped us**: adding IndicQA + BanglaRQA more
than doubled the index (8,315 → 17,568) and moved coverage **not at all** (125 → 125 matched,
corr 0.806 → 0.807). Context coverage was already saturated at ~92%; more of the same shape is
worthless. Lowering `MATCH_THRESHOLD` doesn't rescue it either — unmatched rows sit at *median
cosine 0.374*, nowhere near the 0.75 gate (dropping to 0.50 buys only +31 rows of likely-garbage
matches). **The lever was never "more data." It was "the missing shape."**

### `hishab/bangla-mmlu` — the free substitute for gated BnMMLU
`samanjoy2/BnMMLU` (Alvee's real +0.038 lever) is **gated** — `"gated":"manual"`, confirmed 401 on
the parquet endpoint, no mirror exists. Its ACL 2026 paper claims *"publicly available under
CC BY-SA 4.0, ensuring free accessibility"*, which **contradicts the gate** — usable leverage if
you ever email the author (saman.sarker.joy@gmail.com). **But you don't need it.**
`hishab/bangla-mmlu` (Hishab, the TituLM lab) is **~88K Bengali exam MCQs, ungated, free** — same
exam-bank shape, and it delivered the +0.119 real jump (0.724 → 0.843) that BnMMLU gave Alvee at
a fraction of the size. Schema is `question` + `choices[]` + `answer` as a **letter** (A/B/C/D),
so the letter must be resolved back to the option text (`_MCQ_LETTER_TO_IDX` in the loader).
Also ungated and unexplored: `hishab/titulm-bangla-mmlu` has explicit **Bengali_Grammar** and
**Bengali_Literature** configs — aimed squarely at the 28 still-unmatched vocabulary rows.

### The result: 3 features beat the entire GPU stack
```
matcher alone + xgb        hallu_F1@avg_thr=0.8212   <- 3 features
matcher alone + logreg     hallu_F1@avg_thr=0.8077
matcher alone + gnb        hallu_F1@avg_thr=0.8066
his features alone + logreg hallu_F1@avg_thr=0.7500  <- gemma-4 + SummaC + math solver, 8 features
his + matcher + logreg     hallu_F1@avg_thr=0.8683   <- OOF winner -> real 0.843
combined (fusion) + logreg hallu_F1@avg_thr=0.8662
```
**A TF-IDF lookup plus a string comparison (3 features, CPU-only, ~1 min) outscored the whole
gemma-4/SummaC-NLI/cross-lingual/math-solver stack (22 features, hours of T4).** Every other
approach all night tried to *infer* faithfulness; this one *looks it up*. That's the lesson.

### The GNB collapse — diagnosed and self-resolved
With the 2-source matcher, every gnb config including `MATCHER_FEATURES` collapsed to an
*identical* **0.0290** (`matcher alone + gnb`, `his + matcher + gnb` — same number, the tell).
Cause: `gt_matched` is binary and `gt_agreement` piled ~58% of its mass on exactly 0.5 (the
neutral value for unmatched rows) → near-zero within-class variance → Gaussian likelihood blows
up and saturates the posterior, drowning every other feature. **At 70% coverage the 0.5-spike
shrinks to 30% and the collapse vanishes on its own**: `matcher alone + gnb` went 0.029 → 0.8066.
No code change. If it ever returns, `GaussianNB(var_smoothing=1e-3)` (default `1e-9`) is the fix.
⚠️ This is why `submission_his_matcher_gnb.csv` was briefly garbage — it hardcodes gnb, and gnb
was the one classifier that couldn't eat the matcher features.

### Coverage reached (5 sources, 105,262 indexed QA pairs)
- **Train: 210/299 (70.2%)**, corr(gt_agreement, label) = **0.809**; context 93.9% (corr 0.816),
  no-context **51.5%** (corr **0.798** — nearly as strong as context)
- **Test: 1,556/2,516 (61.8%)** — context 70.5%, **no-context 51.6% (596 of 1,155 previously-dead rows)**
- Coverage rose 68% with **zero** correlation loss (0.806 → 0.809) — no precision/coverage tradeoff

### ⚠️ Phase 2 risk — do not assume this transfers
This works because Phase 1's test set reuses public QA data and we're reading the answer key.
Phase 2's stated source domains are Common Crawl, newspapers, bdlaws, gov sites, ebanglalibrary,
Wikipedia/Banglapedia, **and NCTB textbooks**. If Phase 2 sources fresh text, matcher coverage
could crater — only the NCTB-QA lane has a clear reason to survive. The organizers already scrub
BenHalluEval/idiom-paper samples from Phase 2, so they are demonstrably aware of overlap. **Phase
2 needs all five parquet caches pre-bundled (no internet) AND a fallback plan for near-zero
matcher coverage.**

## (historical) Ground-Truth Source Expansion — as originally built, 2 sources

**Self-critique that led here, worth remembering**: earlier hypothesis-testing for "why does Alvee score higher" only tested what already existed in the codebase (swap XGBoost for GaussianNB, hard-override with *our* 9-idiom dictionary) and concluded "hard overrides don't have enough reach" as a general verdict. That was backwards — it should have been "*our* deterministic sources don't have enough reach, what other ones exist that we haven't looked for at all?" Alvee's answer was a 134K-question exam-bank, not a bigger version of our own rules. **Lesson: when stuck, search for new external ground-truth sources before concluding an approach-class is exhausted.**

### Confirmed-real, confirmed-accessible candidate datasets (found via web search, not assumed)
- **BnMMLU** — 134,375 MCQ question-option pairs across 41 domains (STEM, humanities, social sciences, general knowledge), math content preserved via MathML, ACL 2026 Findings. This is what Alvee's v21 override used. `github.com/samanjoy2/bnmmlu`.
- **NCTB-QA** — large-scale Bangla educational QA built from NCTB (National Curriculum and Textbook Board) content. **Arguably the single most aligned candidate** — NCTB textbooks are explicitly named as a Phase 2 source domain in `some_catches.txt`, so this may matter even more than BnMMLU for this specific competition. `arxiv.org/html/2603.05462v1`.
- **BanglaRQA** — 3,000 context passages, 14,889 QA pairs (factoid/confirmation/list/causal question types, answerable + unanswerable). ACL Findings EMNLP 2022. `github.com/sartajekram419/BanglaRQA`.
- **IndicQA** (`ai4bharat/IndicQA` on HuggingFace) — expert-annotated context+question+answer triples, includes Bengali, CC-BY-4.0. One of the sources the *original organizer reference notebook* (`banglabert-train-m.ipynb`) used, per its own postmortem above.
- **BanglaQuAD** — Bengali open-domain QA dataset, found but not yet evaluated for fit. `arxiv.org/pdf/2410.10229`.
- **TyDi QA Bengali gold-passage** — already sitting locally, zero download cost: `curious_insight/Sample Selection for QA/Datasets/tydiqa_goldp_bengali.csv`. Fastest thing to prototype the matcher against first. (Using this as a *retrieval/lookup* ground-truth source is a legitimately different use than the BenHalluEval training-pair trap — TyDi QA is genuine (passage, question, answer) data, not synthetic LLM-generated hallucination examples with a narrow generation methodology. Don't conflate the two failure modes.)

### What got built (`ground_truth_matcher.py`)
Implemented the plan below almost exactly as scoped, with one change forced by reality:
**BnMMLU turned out to be gated** — its HuggingFace dataset repo (`samanjoy2/BnMMLU`) returns
a real, confirmed 401 on the parquet endpoint (`"gated":"manual"` in the HF API response); the
GitHub repo (`samanjoy2/bnmmlu`) is code/scripts only, no data files. Alvee must have his own
approved HF access or a local copy — we don't. **Not wired up.** Ask him for it directly (see
Next Steps) rather than re-requesting HF access and waiting on manual approval mid-competition.

In its place, **NCTB-QA turned out to be the better find anyway** — confirmed public/ungated
(`ShihabReza/nctb-qa` on HF, ~5MB parquet, ordinary 200 on direct download), 5,812 Bengali QA
pairs built from real NCTB Class 6-9 ICT/Science textbook content (question/answer/evidence
fields, not just MCQ options) — exactly the source domain named in `some_catches.txt`. TyDi
QA (2,503 rows, already local) is indexed alongside it as a general-knowledge supplement.

**Matcher design**: one `GroundTruthMatcher` class, char-ngram (2-4) TF-IDF index over both
sources' questions combined (8,315 total), cosine similarity ≥0.75 = "matched" (conservative,
per the plan below). For matched rows, `_agreement()` compares `response_bn` against the
known answer: numeric answers require an *exact* digit match (including a Bengali
number-word→digit reader reusing `joggota_core.py`'s `_BN_NUMBER_WORDS` list, so spelled-out
"আঠারশ বত্রিশ" is recognized as 1832) — no partial credit for a wrong number that happens to
share surrounding words; non-numeric answers fall back to substring containment, then Jaccard
token overlap. Produces three features (`gt_match_score`, `gt_agreement`, `gt_matched`), kept
in a separate `MATCHER_FEATURES` list rather than folded into `OUR_FEATURES` (see Fusion
Pipeline section above for why — GaussianNB isolation).

**Local validation against the 299 labeled training rows** (step 5 of the plan below, done
before trusting this further): **125/299 rows matched (42% coverage)** — far higher than
Alvee's BnMMLU-only 110/2,516 (~4.4%), because NCTB-QA's textbook-fact style overlaps more of
this competition's factual-QA rows than an MCQ exam bank does. `corr(gt_agreement, label) =
0.806`. Mean `gt_agreement` is **0.937 on true faithful rows vs 0.161 on true hallucinated
rows** — a clean, large separation, not a marginal signal. Manual spot-check of the first 15
matches (printed by the module's own `__main__` block) showed sensible behavior: correct
answers agree, wrong answers (including subtle ones like an off-by-one Ottoman sultan, or a
close-but-wrong mountain height 720m vs the true 670m) correctly disagree. **Test-set
coverage: 961/2,516 rows (38.2%)** — consistent with the train-set rate, not a train-only
artifact. This is now the single highest-coverage deterministic signal in the whole pipeline.

**Kaggle wiring**: `ground_truth_matcher.py` is inlined into `fusion_kaggle.ipynb` as its own
Part 1.5 cell (CPU-only, runs before the GPU lanes) by `fusion_converter.py`, which also
strips the module's `__main__` self-test block first. **Both sources now auto-download** at
runtime if missing (Phase 1 permits internet; Phase 2 will need both pre-bundled instead —
`nctb_qa_train.parquet` and `tydiqa_goldp_bengali.parquet`, both already sitting locally as
of this writing). TyDi QA's auto-download was verified to reconstruct the exact same 2,503
rows as the original local CSV (2,390 train + 113 validation from the upstream
`secondary_task`/gold-passage config, byte-for-byte content match on spot-check) — confirmed
empirically by deleting the local file, forcing the download path, and diffing the result,
not assumed. Filtered/cached to a small per-language parquet so this only happens once.

**`fusion_evaluate.py`** now tests 6 feature-set × 3 model combos (18 total, was 3×3=9) and
always writes three CSVs regardless of which wins OOF: `submission_fusion.csv` (OOF-picked,
same unreliability caveat as before), `submission_his_gnb_baseline.csv` (fixed his-alone+gnb,
our real 0.717 reference), and **`submission_his_matcher_gnb.csv`** (fixed his+matcher+gnb —
same reliable base as the 0.717 run, with the new matcher signal added, still avoiding
`OUR_FEATURES`' binary rule flags that are specifically what broke GaussianNB before).

**Not yet done**: this has not run on Kaggle. All numbers above are local-CV against the
299-row training set only — exactly the kind of number that overfit-threshold-tuning and
OOF-model-selection already burned us on twice tonight (see Leaderboard Status). Strong local
signal is a reason to run it, not a reason to trust it yet.

### Integration plan (as originally scoped — see "What got built" above for what actually happened)
1. **One reusable matcher, not N one-off ones.** Given (`prompt_bn`, `response_bn`), search each dataset's question field for a match; if matched, compare `response_bn` against the dataset's known correct answer. Same shape as the existing math solver (`solve_row` in `fusion_pipeline.py`).
2. **Exact/near-exact matching only, not fuzzy semantic matching, at least for v1.** Alvee's own numbers ("110 high-conf exact-match") suggest deliberate conservatism. A false-positive match injects *wrong* ground truth — worse than having no signal at all.
3. **Feed it as a soft feature into the OOF-validated model, not a hard override.** Deliberate change from Alvee's own math_override, which fires on 0/299 training rows (zero validation coverage) yet directly overrode 136 test rows blind in his notebook — a real, if apparently non-fatal, risk. A feature the model learns to weight, that we can actually measure via OOF CV, is safer than a blind bypass.
4. **Priority order**: NCTB-QA (most aligned with stated Phase 2 domains) → BnMMLU (proven, Alvee's real lever, largest) → BanglaRQA/IndicQA (general factual backup) → TyDi QA local file (fastest to prototype, already on disk).
5. **Validate match precision manually on a small sample before trusting it at scale** — apply the lesson from tonight (in-sample metrics, unvalidated overrides, OOF overfitting all bit us) proactively instead of discovering another version of the same mistake the hard way.

## What Actually Happened

### TigerLLM-9B OOM
Kaggle T4 (16GB) could not fit TigerLLM-9B even at 4-bit quantization. The model loads at ~5.5GB but with tokenizer buffers, CUDA context, and VRAM fragmentation, it consistently OOMed. This killed the LLM judge approach.

### BanglaBERT Failure (CV 0.823 → Actual 0.602)
Attempted as a GPU-free fallback. Fine-tuned locally on CPU using 7.5K pairs extracted from BenHalluEval:
- 5,018 hallucinated (label=0): mostly gibberish, non-Bengali tokens, obvious nonsense
- 2,509 faithful (label=1): 90% clean but 10% mislabeled

**Why it failed**: The BenHalluEval hallucinated samples were TOO easy. They contained English tokens, mixed scripts, and obvious non-answers. BanglaBERT learned to detect "non-Bengali token" as its primary signal — not actual faithfulness. On the diverse competition test set (summarization, reasoning, code-mixed), this shortcut failed catastrophically. CV F1 of 0.823 was a mirage because the 299 training samples came from the same easy-QA distribution.

### Why the Original Notebook's BanglaBERT Worked
The `banglabert-train-m.ipynb` (competition authors) used:
- **81K training pairs** from HuggingFace (30× more data)
- **Qwen 32B judge** to generate hard pseudo-labeled negatives
- **2-seed bagging** (two independent runs, averaged)
- **NLI + SQuAD + IndicQA + BanglaRQA** as additional training sources

We had none of this. Training on only 2.5K easy BenHalluEval pairs taught the model a cheap surface pattern, not real discrimination.

### Fusion pipeline session (this file's biggest update — see "Fusion Pipeline" section above for full detail)
In one very long session: fixed the hallu-F1 metric bug, discovered OOF threshold tuning overfits on 299 rows (proven via a real leaderboard mismatch, not theorized), found and dismissed two teammates' alternative approaches (one risky/miscalibrated, one genuinely strong at 0.777), built a fusion pipeline combining our cross-lingual work with the strong teammate's gemma-4/SummaC/math-solver approach, found/fixed six real bugs across two full Kaggle run attempts (guard-stripping, cell-ordering ×2, missing feature computation, transformers-import-ordering, Jupyter stdout incompatibility), and got two real submissions out of it: 0.717 (his features + gnb) and 0.692 (his features + logreg, despite being the OOF "winner" — see the OOF-model-selection-unreliability note in Leaderboard Status). Neither beat the teammate's solo 0.777, and the OOF comparison honestly showed our combined feature set didn't beat "his features alone" either, so most of tonight's feature-engineering effort didn't demonstrably move the needle over his original approach. What *did* move the needle for him: a BnMMLU exam-answer-key override we didn't know existed until asked directly. Found four more real, accessible ground-truth-source candidates in response (NCTB-QA, BanglaRQA, IndicQA, BanglaQuAD) plus one already sitting locally (TyDi QA gold-passage). That's the scoped next step, not more feature tinkering on the current architecture.

## Next Steps

- **Use `evaluate_source.py` before adding ANY source.** It reports the only number that
  predicts a real gain: *incremental* coverage on no-context rows, plus whether corr holds.
  It has already killed two plausible-looking leads (see below) in 30 seconds each.
- **⚠️ `hishab/titulm-bangla-mmlu` is a MEASURED DEAD END — do not spend time on it.**
  Its `Bengali_Grammar`+`Bengali_Literature` configs (11,421 pairs) look perfect on paper and
  score 13.2% no-context coverage *alone* at corr 0.871 — but incremental gain is **+0 rows**
  (210 → 210). `bangla-mmlu` already covers those exact rows. **The exam-bank shape is now
  saturated, just like the Wikipedia-QA shape was.** Same trap, second time. `researchwithmaisha/bangla-mmlu` appears to mirror hishab's splits — almost certainly +0 too.
- **What's actually left: 89/299 unmatched (81 no-context), and they need a THIRD shape.**
  Task breakdown of the 81: **factual 49, vocabulary 25**, math 4, spelling 2, grammar 1.
  Their best-match cosine is *median 0.476, max 0.748* — genuinely absent from all 105K pairs,
  not near-misses, so threshold tuning cannot rescue them. They're also **60% hallucinated
  (49/81 label=0)**, so they're worth real F1. Two distinct targets:
  - **25 vocabulary rows** — বাগধারা/ভাবার্থ/synonym lookups (`"ধান্ধা" এর ভাবার্থ কী?`,
    `ইংরেজি ভাষায় কম্পিটেন্ট শব্দের অর্থ কী?`). **A QA dataset will never contain these — they need a
    dictionary, not a QA corpus.** Bengali বাগধারা lists, সমার্থক/বিপরীত শব্দ tables, Bengali-English
    glossaries. This is the clearest untried *shape*.
  - **49 factual rows** — heavily current-events/literature (`ডক্টর ইউনূসের নেতৃত্বে অন্তর্বর্তীকালীন সরকার কবে
    শপথ গ্রহণ করে?`, `'আগুনপাখি' উপন্যাসের রচয়িতা কে?`). Post-2024 politics and Bengali literature —
    likely needs a **news/Wikipedia dump**, not a QA dataset. Note this overlaps the organizers'
    stated Phase 2 domains (newspapers, Banglapedia), so it may pay twice.
- **Improve `_agreement()`, not coverage, on already-matched rows.** corr is 0.798-0.816 on
  matched rows, not 1.0 — some matched rows are still scored wrong. Cheap to iterate locally
  (CPU, seconds), and it lifts every matched row at once.
- **Phase 2 hardening** (see the risk box above) — pre-bundle all 5 parquet caches as a Kaggle
  Dataset for the offline requirement, and decide what the pipeline does when matcher coverage
  is near-zero. **The 0.843 depends on a lever that may not exist in Phase 2.** The `his`
  features (gemma-4 + SummaC) are the Phase-2 floor at ~0.717-0.75, and they still work.
- **BnMMLU is no longer needed** — `hishab/bangla-mmlu` substituted for it successfully. Only
  worth chasing if the exam-bank lane needs more depth; the paper's own "freely accessible under
  CC BY-SA 4.0" claim contradicts its gate and is leverage with the author if so.
- **Find the real corpus** — organizer-confirmed to exist in an unfetched Discussion post (see
  "Removed: Mini-RAG" above). Less urgent now that no-context rows have a working signal, but
  still the only route to *external grounding* rather than answer-key lookup — which is exactly
  what Phase 2 will likely require.
- **Watch Discussion for the cached frontier-model outputs** (GPT-4o/Claude/Gemini judgments) — organizer said "not yet, will release, keep watching" — the "strongest single signal" per the welcome note, not usable until released.
- Old ideas below (XGBoost tuning, data augmentation) are lower priority — revisit only if the ground-truth-source expansion plateaus:
  - Per-band threshold calibration (context vs no-context, task type) instead of one global threshold
  - Synonym substitution / word-order shuffling on the 299 training rows to roughly double effective training size

## Key Constraints (from `LEGAL_DOCS.txt` & `some_catches.txt`)

- **Phase 1 permits internet-based resources** — Section 4: "teams may generate predictions using any resources they have access to, provided the same predictions can be reproduced by a code-competition-compliant Phase 2 package." Confirmed this is a plain CSV-upload competition for Phase 1 (Kaggle doesn't execute your notebook to score you), so a Kaggle notebook with Internet ON is fine for Phase 1 even though Phase 2 requires fully offline.
- **Phase 2 GPU**: Single P100 or 2×T4, under 9 hours total runtime, under 50GB disk
- **No internet at inference in Phase 2** — everything must be offline there
- **Open-weight models only** — no paid APIs
- **~5,000 test rows in Phase 2** (2× Phase 1) — inference efficiency matters
- **Phase 2 source domains** (broader than Phase 1):
  - Common Crawl Bengali text
  - Bengali Wikipedia / Banglapedia
  - Government websites (bangladesh.gov.bd, Citizen Charter)
  - ebanglalibrary.com
  - Major Bengali newspapers (last 5 years)
  - bdlaws.minlaw.gov.bd (legislation)
  - NCTB textbooks
- **Methodology papers**: BenHalluEval + LREC 2026 idiom paper — Phase 2 test data contains NO samples from these datasets

## Models Used (all open-weight, Phase 2 compliant)

| Model | Size | Role | Status |
|---|---|---|---|
| gemma-4-12b-it | 12B, 4-bit | Bilingual (bn+en) judge — fusion pipeline | ✅ Real Kaggle-confirmed 0.777 in teammate's solo notebook. Needs a source-build `transformers` install, see gotcha above |
| mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | ~280M | SummaC-ZS windowed NLI — fusion pipeline, replaces the line below | ✅ Working, stronger checkpoint (more training data) than our original |
| mDeBERTa-v3-base-mnli-xnli | ~280M | NLI entailment/contradiction (bn + en) — original `submission_pipeline.py` only | ✅ Working, but superseded by the checkpoint above in the fusion pipeline |
| NLLB-200-distilled-600M | ~600M | bn→en translation for cross-lingual NLI | ✅ Working (verified via smoke test) |
| LaBSE | ~470M | Sentence embeddings + cosine sim | ✅ Working |
| Qwen2.5-1.5B-Instruct | 1.5B | Small LLM Judge (no-context rows, logit-based) — original pipeline only | ✅ Works given `accelerate` installed; superseded by gemma-4 in the fusion pipeline |
| TigerLLM-9B-it | 9B | LLM-as-a-Judge | ❌ OOM on T4 at the time — teammate later proved a 12B model *can* fit via proper 4-bit `BitsAndBytesConfig` + eager attention, so "dead end" was premature, not fundamental |
| BanglaBERT-large | 350M | Cross-encoder classifier | ❌ Made scores worse (0.602) |
| GaussianNB / LogisticRegression | — | Simpler decision-layer alternatives to XGBoost | ⚠️ Tested: GaussianNB much worse (0.23) on our features, LogReg roughly tied XGBoost. Model swap alone isn't what explains the teammate's higher score |
| XGBoost | — | Meta-classifier / feature fusion | ✅ Working |