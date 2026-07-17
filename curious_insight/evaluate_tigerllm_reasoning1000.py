import pandas as pd
import torch
import os
import re
import warnings
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ignore the bitsandbytes casting warnings so progress bar stays clean
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes.autograd._functions")

# Configuration
INPUT_FILE = "/home/bio/Desktop/Thesis-401/Evaluation stage/BanglaHalluEval/Hallucination Generated Answers/reasoning_1000.csv"
OUTPUT_FILE = "/home/bio/Desktop/Thesis-401/Evaluation stage/BanglaHalluEval/TigerLLM/tigerllm_reasoning_1000_eval_results.csv"
MODEL_ID = "md-nishat-008/TigerLLM-9B-it"

def extract_yes_no(text):
    """Normalize the response to strictly 'yes' or 'no' or 'unknown'."""
    t = text.lower().strip()
    if not t: return "unknown"
    if "yes" in t or "হ্যাঁ" in t or t.startswith('y'): return "yes"
    if "no" in t or "না" in t or t.startswith('n'): return "no"
    return "unknown"

def main():
    print(f"Reading input file: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    # Handle resuming — key on integer row index for safety even if `id` is unique
    output_columns = [
        'row_index', 'id', 'source_id', 'question_id', 'hallucination_type',
        'question', 'reasoning_chain_evaluated', 'answer_evaluated',
        'is_hallucinated', 'raw_response'
    ]
    if os.path.exists(OUTPUT_FILE):
        try:
            processed_df = pd.read_csv(OUTPUT_FILE)
            processed_indices = set(processed_df['row_index'].astype(int).tolist())
            print(f"Resuming: found {len(processed_indices)} already processed rows.")
        except Exception as e:
            print(f"Could not read existing output file. Starting fresh. Error: {e}")
            processed_indices = set()
            pd.DataFrame(columns=output_columns).to_csv(OUTPUT_FILE, index=False)
    else:
        processed_indices = set()
        pd.DataFrame(columns=output_columns).to_csv(OUTPUT_FILE, index=False)

    print(f"Loading model: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Define quantization config explicitly for Gemma 3 architecture backward compatibility
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
    )

    print("Starting evaluation...")
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        pass

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating reasoning"):
        if int(index) in processed_indices:
            continue

        row_id = str(row['id'])
        source_id = row.get('source_id', '')
        question_id = row.get('question_id', '')
        hallucination_type = row.get('hallucination_type', '')
        question = row['question']
        chain = row.get('hallucinated_chain', '') or row.get('reasoning', '') or ''
        candidate_answer = row.get('hallucinated_answer', '') or row.get('answer', '') or ''

        # Prompt taken verbatim from the reference OpenAI labeler (label_reasoning_openai.py)
        prompt = (
            "You are an evaluator for Bengali mathematical reasoning tasks.\n"
            "Decide whether the provided reasoning chain is hallucinated relative to the question.\n"
            "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
            "Interpretation: 'yes' means the chain or answer contains incorrect steps or unsupported facts.\n"
            "Provide the answer in English only: yes or no.\n\n"
            f"Question: {question}\n"
            f"Reasoning Chain: {chain}\n"
            f"Answer: {candidate_answer}\n\n"
            "Answer now:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.01,
                do_sample=False
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_part = generated_text[len(prompt):].strip()
        label = extract_yes_no(response_part)

        result_item = {
            'row_index': int(index),
            'id': row_id,
            'source_id': source_id,
            'question_id': question_id,
            'hallucination_type': hallucination_type,
            'question': question,
            'reasoning_chain_evaluated': chain,
            'answer_evaluated': candidate_answer,
            'is_hallucinated': label,
            'raw_response': response_part
        }

        pd.DataFrame([result_item]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        processed_indices.add(int(index))

    print(f"Evaluation complete. Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
