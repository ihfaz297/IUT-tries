"""Run TigerLLM-9B-it once over the three 1000-sample ground-truth datasets
(QA, summarization, reasoning) and write three separate result CSVs.

Loads the model a single time and processes the 3000 rows back-to-back.
Each task uses the prompt template taken VERBATIM from its respective
reference labeler script.
"""

import json
import os
import re
import warnings

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ignore the bitsandbytes casting warnings so progress bar stays clean
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes.autograd._functions")

MODEL_ID = "md-nishat-008/TigerLLM-9B-it"

QA_INPUT = "/home/bio/Desktop/Thesis-401/Evaluation stage/BanglaHalluEval/BanglaHalluEval Datasets/banglahallueval_qa_1000.csv"
SUMM_INPUT = "/home/bio/Desktop/Thesis-401/Evaluation stage/BanglaHalluEval/Sample Selection for Summ/banglahallueval_summarization_dataset_1000.csv"
REASON_INPUT = "/home/bio/Desktop/Thesis-401/Evaluation stage/BanglaHalluEval/Reasoning/somadhan_1000_main_ordered.csv"

OUTPUT_DIR = "/home/bio/Desktop/Thesis-401/Evaluation stage/BanglaHalluEval/TigerLLM/Ground_Truth_Results"
QA_OUTPUT = os.path.join(OUTPUT_DIR, "tigerllm_qa_1000_ground_truth_eval_results.csv")
SUMM_OUTPUT = os.path.join(OUTPUT_DIR, "tigerllm_summarization_1000_ground_truth_eval_results.csv")
REASON_OUTPUT = os.path.join(OUTPUT_DIR, "tigerllm_reasoning_1000_ground_truth_eval_results.csv")


# ---- prompt templates (verbatim from the reference labeler scripts) ----------

QA_PROMPT_TEMPLATE = (
    "You are an evaluator.\n"
    "Decide whether the provided model answer is hallucinated relative to the context and question.\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the answer contains information not supported by the context or contradicts it (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Model answer: {answer}\n\n"
    "Answer now:"
)

SUMM_PROMPT_TEMPLATE = (
    "You are an evaluator.\n"
    "Decide whether the provided model summary is hallucinated relative to the document.\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the summary contains information not supported by the document or contradicts it (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Document: {document}\n"
    "Summary: {summary}\n\n"
    "Answer now:"
)

# Reasoning prompt taken verbatim from
# Evaluation stage/BanglaHalluEval/Reasoning/Evaluation Script/Evaluation Script for hallucinated 1000/evaluate_reasoning.py
# Returns a JSON object: {"is_hallucinated": "Yes"} or {"is_hallucinated": "No"}.
REASON_PROMPT_TEMPLATE = (
    "You are an expert evaluator for Bengali mathematical reasoning tasks.\n"
    "Your task is to determine whether the given hallucinated_chain is hallucinated (i.e., incorrect or fabricated).\n\n"
    "Question: {question}\n"
    "Reasoning Chain: {chain}\n"
    "Answer: {answer}\n\n"
    "Is this hallucinated_chain hallucinated? Respond ONLY with a JSON object like this:\n"
    "{{\"is_hallucinated\": \"Yes\"}} or {{\"is_hallucinated\": \"No\"}}\n"
    "Do not explain your reasoning or output anything else.\n"
)


def extract_yes_no(text):
    """Normalize the response to strictly 'yes' or 'no' or 'unknown'."""
    t = text.lower().strip()
    if not t:
        return "unknown"
    if "yes" in t or "হ্যাঁ" in t or t.startswith("y"):
        return "yes"
    if "no" in t or "না" in t or t.startswith("n"):
        return "no"
    return "unknown"


def extract_json_label(text):
    """Parse {"is_hallucinated": "Yes"|"No"} style outputs (with fallback)."""
    if not text:
        return "unknown"
    # Try a strict JSON parse first
    try:
        obj = json.loads(text.strip())
        val = obj.get("is_hallucinated", "")
        return extract_yes_no(str(val))
    except Exception:
        pass
    # Fallback: regex out the value even if JSON is truncated/extra-text
    m = re.search(r'"is_hallucinated"\s*:\s*"?([A-Za-z]+)"?', text)
    if m:
        return extract_yes_no(m.group(1))
    return extract_yes_no(text)


def load_resume_state(output_path, output_columns):
    """Initialize or read existing output, return set of processed row_indices."""
    if os.path.exists(output_path):
        try:
            done = pd.read_csv(output_path)
            indices = set(done["row_index"].astype(int).tolist())
            print(f"  Resuming {os.path.basename(output_path)}: {len(indices)} already done.")
            return indices
        except Exception as e:
            print(f"  Could not read existing output ({e}); starting fresh.")
    pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)
    return set()


def generate_response(model, tokenizer, prompt, max_new_tokens=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.01,
            do_sample=False,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()


def generate_label(model, tokenizer, prompt):
    response_part = generate_response(model, tokenizer, prompt, max_new_tokens=10)
    return extract_yes_no(response_part), response_part


# ---- per-task drivers --------------------------------------------------------

def run_qa(model, tokenizer):
    print(f"\n=== QA: {QA_INPUT} ===")
    df = pd.read_csv(QA_INPUT)
    columns = ["row_index", "id", "context", "question", "answer_evaluated", "is_hallucinated", "raw_response"]
    processed = load_resume_state(QA_OUTPUT, columns)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="QA"):
        if int(index) in processed:
            continue

        context = row.get("context", "") or ""
        question = row.get("question", "") or ""
        answer = (
            row.get("correct_answer")
            or row.get("correct answer")
            or row.get("model_answer")
            or row.get("answer")
            or ""
        )

        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question, answer=answer)
        label, raw = generate_label(model, tokenizer, prompt)

        pd.DataFrame([{
            "row_index": int(index),
            "id": row.get("id", ""),
            "context": context,
            "question": question,
            "answer_evaluated": answer,
            "is_hallucinated": label,
            "raw_response": raw,
        }]).to_csv(QA_OUTPUT, mode="a", header=False, index=False)
        processed.add(int(index))

    print(f"  Saved to {QA_OUTPUT}")


def run_summarization(model, tokenizer):
    print(f"\n=== Summarization: {SUMM_INPUT} ===")
    df = pd.read_csv(SUMM_INPUT)
    columns = ["row_index", "id", "document", "summary_evaluated", "is_hallucinated", "raw_response"]
    processed = load_resume_state(SUMM_OUTPUT, columns)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Summ"):
        if int(index) in processed:
            continue

        # The 1000-sample summarization dataset stores the document in the 'question' field
        # (per the reference labeler's comment).
        document = row.get("question", "") or ""
        summary = row.get("summary", "") or ""

        prompt = SUMM_PROMPT_TEMPLATE.format(document=document, summary=summary)
        label, raw = generate_label(model, tokenizer, prompt)

        pd.DataFrame([{
            "row_index": int(index),
            "id": row.get("id", ""),
            "document": document,
            "summary_evaluated": summary,
            "is_hallucinated": label,
            "raw_response": raw,
        }]).to_csv(SUMM_OUTPUT, mode="a", header=False, index=False)
        processed.add(int(index))

    print(f"  Saved to {SUMM_OUTPUT}")


def run_reasoning(model, tokenizer):
    print(f"\n=== Reasoning: {REASON_INPUT} ===")
    df = pd.read_csv(REASON_INPUT)
    columns = ["row_index", "question_id", "question", "answer_evaluated", "is_hallucinated", "raw_response"]
    processed = load_resume_state(REASON_OUTPUT, columns)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Reasoning"):
        if int(index) in processed:
            continue

        question = row.get("question", "") or ""
        answer = row.get("answer", "") or ""
        # The ground-truth reasoning dataset has no separate `hallucinated_chain` column,
        # so we feed the same `answer` text in as the chain (it IS the full chain).
        chain = answer

        prompt = REASON_PROMPT_TEMPLATE.format(question=question, chain=chain, answer=answer)
        # JSON output needs more tokens than the yes/no prompts.
        raw = generate_response(model, tokenizer, prompt, max_new_tokens=40)
        label = extract_json_label(raw)

        pd.DataFrame([{
            "row_index": int(index),
            "question_id": row.get("question_id", ""),
            "question": question,
            "answer_evaluated": answer,
            "is_hallucinated": label,
            "raw_response": raw,
        }]).to_csv(REASON_OUTPUT, mode="a", header=False, index=False)
        processed.add(int(index))

    print(f"  Saved to {REASON_OUTPUT}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in (QA_INPUT, SUMM_INPUT, REASON_INPUT):
        if not os.path.exists(path):
            raise SystemExit(f"Input file not found: {path}")

    print(f"Loading model: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 8-bit quantization (Gemma 3 architecture)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    print("Starting evaluation...")
    run_qa(model, tokenizer)
    run_summarization(model, tokenizer)
    run_reasoning(model, tokenizer)

    print("\nAll three tasks complete.")
    print(f"  QA:            {QA_OUTPUT}")
    print(f"  Summarization: {SUMM_OUTPUT}")
    print(f"  Reasoning:     {REASON_OUTPUT}")


if __name__ == "__main__":
    main()
