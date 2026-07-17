import pandas as pd
import requests
import json
import os
from tqdm import tqdm

# Configuration
INPUT_FILE = "tydiqa_goldp_bengali.csv"
OUTPUT_FILE = "gemma_answers_bengali.csv"
MODEL_NAME = "gemma2:27b"
OLLAMA_URL = "http://localhost:11434"

def get_ollama_response(question):
    """
    Sends a question to the Ollama Chat API and returns a short, precise response.
    """
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise QA assistant. Answer only in 1-2 Bengali words. No explanation. No meta-talk. Just the direct answer."
            },
            {
                "role": "user",
                "content": f"প্রশ্ন: {question}"
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            answer = response.json().get("message", {}).get("content", "").strip()
            # Post-process: Take first 2 words max to be safe
            import re
            # Split by whitespace, take first 2
            words = answer.split()
            if len(words) > 2:
                answer = " ".join(words[:2])
            return answer
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Exception: {str(e)}"

def main():
    print(f"Reading input file: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Handle resuming if the output file already exists
    if os.path.exists(OUTPUT_FILE):
        try:
            processed_df = pd.read_csv(OUTPUT_FILE)
            processed_ids = set(processed_df['serial_id'].astype(str).tolist())
            print(f"Resuming: found {len(processed_ids)} already processed items.")
        except Exception as e:
            print(f"Could not read existing output file, starting fresh. Error: {e}")
            processed_ids = set()
            pd.DataFrame(columns=['serial_id', 'question', 'model_answer', 'context']).to_csv(OUTPUT_FILE, index=False)
    else:
        processed_ids = set()
        # Initialize output file with headers
        pd.DataFrame(columns=['serial_id', 'question', 'model_answer', 'context']).to_csv(OUTPUT_FILE, index=False)

    print(f"Starting zero-shot evaluation with model: {MODEL_NAME}")
    
    # Use a list to collect results to append in batches or one-by-one
    # Appending one-by-one ensures minimal data loss if interrupted
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        serial_id = str(row['id'])
        question = row['question']
        context = row['context']
        
        if serial_id in processed_ids:
            continue
            
        # Zero-shot evaluation: use only the question
        answer = get_ollama_response(question)
        
        result_item = {
            'serial_id': serial_id,
            'question': question,
            'model_answer': answer,
            'context': context
        }
        
        # Append single row to CSV
        pd.DataFrame([result_item]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        processed_ids.add(serial_id)

    print(f"Evaluation complete. Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
