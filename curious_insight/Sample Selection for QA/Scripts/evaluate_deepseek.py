import pandas as pd
import requests
import json
import os
from tqdm import tqdm

# Configuration
INPUT_FILE = "/home/bio/Desktop/Thesis-401/tydiqa_goldp_bengali.csv"
OUTPUT_FILE = "/home/bio/Desktop/Thesis-401/deepseek_answers_bengali.csv"
MODEL_NAME = "deepseek-r1:14b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def get_ollama_response(question):
    """
    Sends a question to the Ollama Chat API and returns a short, precise response.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise QA assistant. Answer the user question in 1-3 Bengali words maximum. Do NOT include the question, do NOT include any introductory text (like 'The answer is' or 'উত্তর:'), and do NOT include any explanation. Just provide the direct answer as concisely as possible."
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
        response = requests.post(url, json=payload, timeout=90)
        if response.status_code == 200:
            answer = response.json().get("message", {}).get("content", "").strip()
            
            # DeepSeek R1 specific: Remove <think> blocks
            import re
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            
            # Remove common prefixes the model might still add
            answer = re.sub(r'^(উত্তর|Answer|The answer is|সঠিক উত্তরটি হলো|সঠিক উত্তর|উঃ)[:\s\-]*', '', answer, flags=re.IGNORECASE).strip()
            
            # Post-process: Take first 3 words max to be safe but not too aggressive
            words = answer.split()
            if len(words) > 4: # Allow up to 4 words for Bengali compounds
                answer = " ".join(words[:4])
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
