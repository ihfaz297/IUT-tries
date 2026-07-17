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
BASE_DIR = "/home/bio/Desktop/Thesis-401/Latest/BanglaHalluEval"
INPUT_FILE = os.path.join(BASE_DIR, "Hallucination Generated Answers/codemix_4000.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "tigerllm_codemix_4000_eval_results.csv")
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

    # utf-8-sig strips the BOM present on the id column of codemix_4000.csv
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    # Handle resuming
    if os.path.exists(OUTPUT_FILE):
        try:
            processed_df = pd.read_csv(OUTPUT_FILE)
            processed_ids = set(processed_df['id'].astype(str).tolist())
            print(f"Resuming: found {len(processed_ids)} already processed items.")
        except Exception as e:
            print(f"Could not read existing output file. Starting fresh. Error: {e}")
            processed_ids = set()
            pd.DataFrame(columns=['id', 'question', 'context', 'answer_evaluated', 'is_hallucinated', 'raw_response']).to_csv(OUTPUT_FILE, index=False)
    else:
        processed_ids = set()
        pd.DataFrame(columns=['id', 'question', 'context', 'answer_evaluated', 'is_hallucinated', 'raw_response']).to_csv(OUTPUT_FILE, index=False)

    print(f"Loading model: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Define quantization config explicitely for Gemma 3 architecture backward compatibility
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

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating answers"):
        row_id = str(row['id'])
        if row_id in processed_ids:
            continue

        context = row['codemix_context']
        question = row['codemix_question']
        candidate_answer = row['hallucinated_answer']

        # Creating a neutral prompt
        prompt = (
            "You are an expert evaluator. Carefully read the Context, the Question, and the Provided Answer.\n"
            "Decide whether the Provided Answer contains factual errors, unsupported claims, or hallucinates "
            "information not present in the Context regarding the Question.\n"
            "Reply strictly with a single word: 'yes' (if the answer is hallucinated/incorrect) or 'no' (if it is accurate and supported).\n\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Provided Answer: {candidate_answer}\n\n"
            "Answer:"
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
            'id': row_id,
            'question': question,
            'context': context,
            'answer_evaluated': candidate_answer,
            'is_hallucinated': label,
            'raw_response': response_part
        }

        pd.DataFrame([result_item]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        processed_ids.add(row_id)

    print(f"Evaluation complete. Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
