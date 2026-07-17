import pandas as pd
import requests
import json
import os
from tqdm import tqdm

INPUT_FILE = "tydiqa_goldp_bengali.csv"
OUTPUT_FILE = "gemma_answers_bengali.csv"
MODEL_NAME = "gemma2:9b"
OLLAMA_URL = "http://localhost:11434"

def get_ollama_response(question):
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a precise QA assistant. Answer only in 1-2 Bengali words."},
            {"role": "user", "content": f"প্রশ্ন: {question}"}
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "").strip()
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Exception: {str(e)}"

def main():
    print(f"Reading {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    if os.path.exists(OUTPUT_FILE):
        processed_df = pd.read_csv(OUTPUT_FILE)
        processed_ids = set(processed_df['serial_id'].astype(str).tolist())
    else:
        processed_ids = set()
        pd.DataFrame(columns=['serial_id', 'question', 'model_answer', 'context']).to_csv(OUTPUT_FILE, index=False)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sid = str(row['id'])
        if sid in processed_ids: continue
        answer = get_ollama_response(row['question'])
        pd.DataFrame([{'serial_id': sid, 'question': row['question'], 'model_answer': answer, 'context': row['context']}]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

if __name__ == '__main__':
    main()
