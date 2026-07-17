# BanglaHalluEval — Evaluation Implementation Log

This document provides a comprehensive account of the hallucination evaluation pipeline implemented for the BanglaHalluEval benchmark. It covers infrastructure setup, dataset details, evaluation methodology, scripts, results, and key implementation decisions. This is intended to support the implementation section of the associated research paper.

---

## 1. Overview

The evaluation pipeline assesses hallucination in Bengali NLP outputs across three task domains:

| Domain | Candidate Set | Ground Truth Set |
|--------|--------------|-----------------|
| Summarization | 3,000 hallucinated summaries | 1,000 reference summaries |
| Reasoning | 1,000 hallucinated reasoning chains | 1,000 ground truth solutions |
| Code-Mixed QA | 4,000 hallucinated answers | 1,000 ground truth answers |

Primary evaluations were conducted using **DeepSeek-R1 14B** as the judge model, served locally via **Ollama** on cloud GPU instances (RunPod). In the current codebase, an additional local evaluation track uses **GPT-4.1-mini** via the OpenAI API to label the Code-Mixed QA candidate and ground-truth sets. All judge models classify each item as hallucinated (`yes`) or not (`no`).

---

## 2. Infrastructure

### 2.1 Compute Environment

All evaluation runs were executed on **RunPod GPU cloud** instances. Two pods were used in parallel to process different tasks simultaneously.

| Pod | GPU | VRAM | Tasks Run |
|-----|-----|------|-----------|
| `general_bronze_toad` | RTX A6000 | 48 GB | Summarization GT (1000), Reasoning Candidates (1000) |
| `select_indigo_termite` | A40 | 48 GB | Summarization Candidates (3000), Reasoning Main (1000) |

### 2.2 Model Serving

The judge model was served using **Ollama** running inside each pod:

```
Model:   deepseek-r1:14b
Size:    ~9.0 GB
Server:  http://localhost:11434
API:     /api/generate  (POST, JSON)
```

Ollama was started as a background process (`ollama serve &`) and the model was pre-pulled before evaluation began (`ollama pull deepseek-r1:14b`).

### 2.3 Session Persistence with tmux

To prevent evaluation runs from being interrupted by terminal disconnections or device shutdowns, all long-running scripts were launched inside **tmux** sessions:

```bash
tmux new -s <session-name>
python3 scripts/<eval_script>.py
# Ctrl+B then D  →  detach (script keeps running)
tmux attach -t <session-name>  # reattach later
```

This was critical given that each full evaluation run takes several hours and pods were accessed remotely over unstable connections.

### 2.4 Local Evaluation (Windows + OpenAI API)

For the Code-Mixed QA re-evaluation with GPT-4.1-mini, scripts were executed locally on Windows using a Python venv. The API key is loaded from a `.env` file (`OPENAI_API_KEY`). The OpenAI Python SDK is used through the Responses API with deterministic settings.

---

## 3. Judge Model: DeepSeek-R1 14B

DeepSeek-R1 is a reasoning-focused language model that produces a chain-of-thought inside `<think>...</think>` tags before giving its final answer. This required special handling during response parsing.

### 3.1 API Call Configuration

```python
payload = {
    "model": "deepseek-r1:14b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "num_ctx": 8192,
        "num_predict": 2048,
        "temperature": 0,
    }
}
```

Key settings:
- `num_predict: 2048` — required to allow the full `<think>` chain to complete before the final answer token
- `temperature: 0` — deterministic output for reproducibility
- `format: "json"` was **intentionally omitted** — using it with DeepSeek-R1 caused empty responses because the JSON grammar constraint conflicted with the model's `<think>` tag output format

### 3.2 Response Parsing

After receiving the raw response, the think chain is stripped and the final yes/no token is extracted:

```python
import re

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

For reasoning tasks that return JSON (`{"is_hallucinated": "Yes"}`), additional JSON parsing with fallback regex was applied:

```python
cleaned = strip_think(raw_response)
try:
    is_hallucinated = json.loads(cleaned).get("is_hallucinated", "")
except Exception:
    # fallback: regex search for Yes/No
    m = re.search(r'"is_hallucinated"\s*:\s*"(Yes|No)"', raw_response, re.IGNORECASE)
    is_hallucinated = m.group(1) if m else ""
```

## 3.3 Judge Model: GPT-4.1-mini (OpenAI API)

GPT-4.1-mini is used for local Code-Mixed QA labeling. The prompt is identical in structure to the DeepSeek codemix prompt but executed via the OpenAI Responses API. The response is normalized to `yes`/`no`, with `unknown` retained when the model output is ambiguous.

**API call configuration (local scripts):**
```python
request_kwargs = {
    "model": "gpt-4.1-mini",
    "input": [{"role": "user", "content": prompt}],
    "max_output_tokens": 16,
    "temperature": 0,
}
response = client.responses.create(**request_kwargs)
label = response.output_text
```

---

## 4. Datasets and Evaluation Design

### 4.1 Summarization (Bengali)

**Candidate set** (`summarization_3000_corrected.csv`): 3,000 hallucinated summaries generated from 1,000 source documents, covering three hallucination patterns:

| Pattern | Count |
|---------|-------|
| Intrinsic | 1,000 |
| Non-factual | 1,000 |
| Factual Contradiction | 1,000 |

The ID format encodes the pattern: `{doc_id}::Intrinsic`, `{doc_id}::Non-factual`, `{doc_id}::Factual Contradiction`.

**Ground truth set** (`banglahallueval_summarization_dataset_1000.csv`): 1,000 human-written reference summaries. Expected result: mostly `no` (faithful summaries should not be flagged as hallucinated). Serves as a calibration/sanity check.

**Prompt used:**
```
You are an evaluator.
Decide whether the provided model summary is hallucinated relative to the document.
Only reply with a single word: yes or no.
Interpretation: 'yes' means the summary contains information not supported by the
document or contradicts it (hallucinated).
Provide the answer in English only.

Document: {document}
Summary: {summary}

Answer now (yes or no):
```

### 4.2 Reasoning (Bengali Mathematical)

**Candidate set** (`reasoning_1000.csv`): 1,000 hallucinated reasoning chains generated from Bengali math problems. Five error types, 200 samples each:

| Error Type | Count |
|-----------|-------|
| `arithmetic_slip` | 200 |
| `formula_misapplication` | 200 |
| `invalid_deduction` | 200 |
| `hallucinated_intermediate_fact` | 200 |
| `semantic_drift` | 200 |

**Ground truth set** (`somadhan_1000_main.csv`): 1,000 correct solutions to the same math problems. Serves as a negative control — correct solutions should predominantly return `no`.

**Prompt used (strictly maintained from reference implementation):**
```
You are an expert evaluator for Bengali mathematical reasoning tasks.
Your task is to determine whether the given hallucinated_chain is hallucinated
(i.e., incorrect or fabricated).

Question: {question}
Reasoning Chain: {hallucinated_chain}
Answer: {hallucinated_answer}

Is this hallucinated_chain hallucinated? Respond ONLY with a JSON object like this:
{"is_hallucinated": "Yes"} or {"is_hallucinated": "No"}
Do not explain your reasoning or output anything else.
```

For the ground truth set, `answer` was used in place of `hallucinated_chain` and `hallucinated_answer` since no separate chain exists.

### 4.3 Code-Mixed QA (Bengali-English)

**Candidate set** (`codemix_4000.csv`): 4,000 hallucinated answers to code-mixed (Bengali-English) questions, covering four patterns:

| Pattern | Count |
|---------|-------|
| `factualness` | 1,000 |
| `comprehension` | 1,000 |
| `specificity` | 1,000 |
| `inference` | 1,000 |

**Ground truth set** (`codemix_1000.csv`): 1,000 correct answers to code-mixed questions.

**Prompt used:**
```
You are an evaluator.
Decide whether the provided model answer is hallucinated relative to the context and question.
Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.
Interpretation: 'yes' means the answer contains information not supported by the context
or contradicts it (hallucinated).
Provide the answer in English only: yes or no.

Context: {codemix_context}
Question: {codemix_question}
Model answer: {answer}

Answer now:
```

---

## 5. Evaluation Scripts

All scripts are in the `scripts/` directory and follow a consistent design:

| Script | Task | Input | Output |
|--------|------|-------|--------|
| `evaluate_deepseek_ollama.py` | Summarization candidates | `summarization_3000_corrected.csv` | `summ_3000_deepseek_hallu.csv` |
| `evaluate_deepseek_summ_gt.py` | Summarization GT | `banglahallueval_summarization_dataset_1000.csv` | `summ_gt_1000_deepseek.csv` |
| `evaluate_reasoning_deepseek_candidates.py` | Reasoning candidates | `reasoning_1000.csv` | `reasoning_1000_candidates_deepseek.csv` |
| `evaluate_reasoning_deepseek_main.py` | Reasoning GT | `somadhan_1000_main.csv` | `reasoning_main_1000_deepseek.csv` |
| `evaluate_codemix_deepseek_candidates.py` | Codemix candidates | `codemix_4000.csv` | `codemix_4000_candidates_deepseek.csv` |
| `evaluate_codemix_deepseek_main.py` | Codemix GT | `codemix_1000.csv` | `codemix_1000_main_deepseek.csv` |
| `label_codemix_hallucinations_gpt_4_1_mini.py` | Codemix candidates (GPT-4.1-mini) | `codemix_4000.csv` | `Hallucination Generated Answers/codemix_4000_gpt4_1_mini_labeled.csv` |
| `label_codemix_main_gpt_4_1_mini.py` | Codemix GT (GPT-4.1-mini) | `codemix_1000.csv` | `Codemix/Main dataset/codemix_1000_gpt4_1_mini_labeled.csv` |

### 5.1 Resume / Checkpoint Logic

All scripts implement row-level checkpointing. On each run, the script reads the output CSV (if it exists), collects all IDs where `is_hallucinated` is already `yes` or `no`, and skips those rows:

```python
completed_ids: set = set()
if out.exists():
    for row in csv.DictReader(open(out)):
        sid = row.get("id")
        if sid and row.get("is_hallucinated") in ("yes", "no"):
            completed_ids.add(sid)

# Then in the main loop:
if sid in completed_ids:
    continue
```

Results are appended after each row, so interruption at any point loses at most one sample. Re-running the same command automatically resumes from where it stopped.

---

## 6. Evaluation Results

### 6.1 Summarization

| Dataset | Total | Yes (Hallucinated) | No (Not Hallucinated) | Unknown |
|---------|-------|-------------------|----------------------|---------|
| Candidates (3,000) | 3,005* | 2,802 (93.2%) | 202 (6.7%) | 1 |
| Ground Truth (1,000) | 1,000 | 176 (17.6%) | 824 (82.4%) | 0 |

*3,005 due to minor duplication from interrupted runs — deduplicated to 3,000 unique rows in post-processing.

**Key observation:** The high `yes` rate (93.2%) on candidates confirms the hallucination generation pipeline successfully produced unfaithful summaries. The GT `yes` rate (17.6%) represents the model's false positive rate on faithful summaries.

### 6.2 Reasoning

| Dataset | Total | Yes (Hallucinated) | No (Not Hallucinated) | Unknown |
|---------|-------|-------------------|----------------------|---------|
| Candidates (1,000) | 1,000 | 814 (81.4%) | 159 (15.9%) | 27 (2.7%) |
| Ground Truth (1,000) | 1,000 | 196 (19.6%) | 789 (78.9%) | 15 (1.5%) |

**Key observation:** 81.4% detection rate on hallucinated reasoning chains. The GT false positive rate of 19.6% is higher than summarization, reflecting the difficulty of evaluating mathematical Bengali text.

### 6.3 Code-Mixed QA

**DeepSeek-R1 (Ollama):**

| Dataset | Total | Yes (Hallucinated) | No (Not Hallucinated) | Unknown |
|---------|-------|-------------------|----------------------|---------|
| Ground Truth (1,000) | 1,000 | 401 (40.1%) | 596 (59.6%) | 3 (0.3%) |
| Candidates (4,000) | 4,000 | 3,439 (85.98%) | 551 (13.78%) | 10 (0.25%) |

**GPT-4.1-mini (OpenAI API):**

| Dataset | Total | Yes (Hallucinated) | No (Not Hallucinated) | Unknown |
|---------|-------|-------------------|----------------------|---------|
| Candidates (4,000) | 4,000 | 3,186 (79.65%) | 786 (19.65%) | 28 (0.70%) |
| Ground Truth (1,000) | 1,000 | 240 (24.0%) | 760 (76.0%) | 0 (0.0%) |

---

## 7. Implementation Challenges and Solutions

### 7.1 Cloud Credit Exhaustion

**Problem:** RunPod pods were automatically paused mid-run when account credits ran out, closing all active connections and terminating foreground processes.

**Solution:** Scripts were designed with row-level checkpointing from the start. When credits were restored and the pod restarted, re-running the same command automatically resumed from the last completed row. In the summarization run, 1,609 of 3,000 rows had been completed before the interruption, saving approximately 54% of the compute that would otherwise need to be redone.

### 7.2 Terminal Disconnection

**Problem:** Closing the browser tab or losing internet connection killed foreground processes on the pod.

**Solution:** All subsequent runs were launched inside **tmux** sessions. A tmux session persists independently of the terminal connection, allowing scripts to run unattended. The workflow:
1. `tmux new -s <name>` — create session
2. Launch script
3. `Ctrl+B D` or close tab — detach safely
4. `tmux attach -t <name>` — reattach from any terminal later

### 7.3 DeepSeek-R1 JSON Format Conflict

**Problem:** Using `"format": "json"` in the Ollama payload caused empty responses from DeepSeek-R1. The model generates a `<think>...</think>` block before the final answer; the JSON grammar constraint interfered with this output structure.

**Solution:** Removed `"format": "json"` from the payload. Instead, `num_predict` was increased to 2,048 to ensure the full think chain completed, and the final answer was extracted by stripping think tags and parsing the remaining text.

### 7.4 Output File Duplication

**Problem:** Scripts interrupted and restarted multiple times caused duplicate rows in output CSVs (e.g., `summ_3000_deepseek_hallu.csv` had 3,005 rows instead of 3,000).

**Solution:** Post-processing deduplication:
```python
df = pd.read_csv(output_file)
df = df[df["is_hallucinated"].isin(["yes", "no"])]
df = df.drop_duplicates(subset=["id"], keep="last")
df.to_csv(output_file, index=False)
```

### 7.5 Python Environment on Pod

**Problem:** Fresh RunPod pods did not have Python, pip, or required packages installed.

**Solution:** One-time setup command per pod:
```bash
apt-get update && apt-get install -y git python3 python3-pip tmux
pip3 install pandas requests tqdm python-dotenv --break-system-packages
```

### 7.6 Git Authentication on Remote Pods

**Problem:** Git push from pods was rejected due to missing credentials or insufficient collaborator permissions.

**Solution:** Used Personal Access Tokens (PAT) with `repo` scope embedded in the remote URL:
```bash
git remote set-url origin https://USERNAME:TOKEN@github.com/AASani29/BanglaHalluEval.git
```

When outbound TCP was blocked (preventing SCP), the GitHub REST API was used directly from Python to upload files via `PUT /contents/{path}`.

---

## 8. Post-Processing Notes

The following files require deduplication before use in analysis:

| File | Current Rows | Expected | Action |
|------|-------------|----------|--------|
| `summ_3000_deepseek_hallu.csv` | 3,005 | 3,000 | Deduplicate on `id`, keep last |

The following files have `unknown` labels that should be reviewed or re-evaluated:

| File | Unknown Count | % of Total |
|------|--------------|------------|
| `reasoning_1000_candidates_deepseek.csv` | 27 | 2.7% |
| `reasoning_main_1000_deepseek.csv` | 15 | 1.5% |
| `codemix_1000_main_deepseek.csv` | 3 | 0.3% |
| `Hallucination Generated Answers/codemix_4000_gpt4_1_mini_labeled.csv` | 28 | 0.7% |

---

## 9. Remaining Work

| Task | Status |
|------|--------|
| Codemix candidates evaluation (4,000) | Completed (DeepSeek-R1 + GPT-4.1-mini) |
| Deduplication of `summ_3000_deepseek_hallu.csv` | Pending |
| Re-evaluation of `unknown` rows | Optional (DeepSeek + GPT-4.1-mini) |
| Pull all pushed results to local | Pending after each push |
