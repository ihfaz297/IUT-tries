# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**অলীকবচন** — Bengali LLM Hallucination Detection Challenge (Datathon 2.0, IUT / IPD / Brain Lab).

Binary classification: given a Bengali prompt and a candidate response (plus an optional context passage), predict whether the response is **faithful (label=1)** or **hallucinated (label=0)**.

**Metric — confirmed, not assumed**: LEGAL_DOCS.txt Section 7 states "Primary metric: binary F1 on the HALLUCINATED class (label = 0)." The Overview tab separately says "macro-F1" — these disagree, and the org has an open Discussion thread about it, but Section 7's wording is the more authoritative "Scoring, Ties, and Disputes" section, and a teammate's independently-built notebook (`phase2-final.ipynb`) reached the same conclusion in its own comments. Treat hallucinated-class F1 as primary. **This was a live bug for most of the competition**: `sklearn.metrics.f1_score(y, pred)` defaults to `pos_label=1`, which is the *faithful* class here — every CV number logged before the fix (including the BanglaBERT 0.823 postmortem below) was silently measuring the wrong class. Fixed in `submission_pipeline.py`/`fast_cv.py` by explicitly passing `pos_label=0`.

Two-phase competition:
- **Phase 1**: Submit prediction CSVs to Kaggle. Public + private leaderboard scoring.
- **Phase 2**: Top 30 teams submit a runnable solution package. Runs **offline in a Kaggle kernel — open-weight models only, no paid APIs**. Build Phase 2-compliant from day one.

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
  combined) × (GaussianNB / LogisticRegression / XGBoost) — picks whichever wins on OOF
  hallu-F1 rather than assuming fusion is automatically better. Threshold is the *average of
  each fold's own held-out best threshold*, not a single value grid-searched against the full
  OOF pool (see the overfitting lesson below — this is a direct mitigation for it).

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
| `fusion_pipeline.py` | **Current best-track pipeline** — combines teammate's gemma-4/SummaC-NLI/math-solver with our cross-lingual/lexical features. See "Fusion Pipeline" section above |
| `fusion_evaluate.py` | Honest OOF CV comparison (his features / ours / combined × 3 model types) + final submission generation, reused from pickles `fusion_pipeline.py` writes |
| `fusion_converter.py` | Assembles `joggota_core.py` + `submission_pipeline.py` + `fusion_pipeline.py` + `fusion_evaluate.py` → `fusion_kaggle.ipynb` |
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

- **Our real, Kaggle-confirmed score (cross-lingual + response-quality features, corrected metric): 0.636** hallu-F1. Matches the un-tuned @0.5-threshold CV number exactly — see the overfitting lesson below.
- **Teammate Alvee's real, Kaggle-confirmed score: 0.777** (`phase2-final.ipynb` — gemma-4-12b-it judge + SummaC NLI + math solver + GaussianNB). Currently the team's best number. Fusion pipeline (above) is the attempt to combine both.
- **Old score, superseded**: 0.602 (BanglaBERT experiment, see postmortem below) — do not treat as current baseline, 0.636 already recovered from it.
- **No-context rows are the primary bottleneck, quantified**: local OOF testing showed **0.787 hallu-F1 on context rows vs 0.500 on no-context rows** — barely better than chance. Root cause: our two highest-importance features (`novel_char_ratio` at 0.12 importance, `context_containment` at 0.11) both structurally return 0 on no-context rows (nothing to compare the response against). This is *why* the fusion pipeline's SummaC lane was extended to score every row using the prompt as a fallback premise instead of leaving no-context rows blank.
- **Train/test distribution mismatch**: train is 43.5% context rows, test is 54.1% — an 10.6pp gap. Worth remembering when trusting CV numbers.
- **⚠️ OOF threshold tuning overfits on 299 rows — proven, not theoretical.** Grid-searching a single global threshold against pooled OOF predictions found 0.78 (hallu-F1 0.7139 in validation) — but the *actual* submitted leaderboard score was 0.636, exactly matching the untuned @0.5 baseline. The "improvement" was fitting noise in a small validation pool, not a real gain. `fusion_evaluate.py` mitigates this by averaging each fold's own independently-found best threshold instead of searching the pooled OOF set once — still imperfect with this little data, but structurally less prone to the same failure.
- **Real feature importances** (from the actual Kaggle run, not local CV): `novel_char_ratio` (0.120) and `context_containment` (0.112) — both cheap, lexical, non-model features — outweighed the entire cross-lingual NLI+translation stack combined (~0.122) and roughly tied the whole Qwen judge (0.067) contribution. The expensive multi-model machinery was real but underpowered relative to simple lexical heuristics; the fusion pipeline's bet is that gemma-4's much larger judge changes this balance.

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
In one long session: fixed the hallu-F1 metric bug, discovered OOF threshold tuning overfits on 299 rows (proven via a real leaderboard mismatch, not theorized), found and dismissed two teammates' alternative approaches (one risky/miscalibrated, one genuinely strong at 0.777), built a fusion pipeline combining our cross-lingual work with the strong teammate's gemma-4/SummaC/math-solver approach, and found/fixed four real bugs in that fusion build (see "Bugs found and fixed" under Fusion Pipeline). Also fixed two real bugs in the deterministic math rule (substring-match misclassification, missing spelled-out-number/MCQ handling — see Deterministic Rules above). Status at end of session: fusion notebook fixed and verified, pending a fresh full Kaggle run (session had disconnected mid-run) — not yet submitted.

## Next Steps

- **Find the real corpus** — organizer-confirmed to exist in an unfetched Discussion post (see "Removed: Mini-RAG" above). Highest-leverage lever specifically for no-context rows, which have no other source of external grounding right now.
- **Run the fusion pipeline to completion and submit** — was interrupted by a Kaggle session disconnect; notebook is fixed, needs a fresh top-to-bottom run.
- **Watch Discussion for the cached frontier-model outputs** (GPT-4o/Claude/Gemini judgments) — organizer said "not yet, will release, keep watching" — the "strongest single signal" per the welcome note, not usable until released.
- Old ideas below (XGBoost tuning, data augmentation) are lower priority now that the fusion pipeline exists — revisit only if fusion plateaus:
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