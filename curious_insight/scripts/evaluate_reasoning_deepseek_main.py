#!/usr/bin/env python3
"""Evaluate main ground truth reasoning dataset using deepseek-r1:14b via Ollama.

Input:  Reasoning/1000_main_dataset/somadhan_1000_main.csv
Output: Reasoning/Results/reasoning_main_1000_deepseek.csv

Usage:
    python3 scripts/evaluate_reasoning_deepseek_main.py
"""

import pandas as pd
import json
import requests
import os
import re
from tqdm import tqdm

INPUT_FILE  = "Reasoning/1000_main_dataset/somadhan_1000_main.csv"
OUTPUT_FILE = "Reasoning/Results/reasoning_main_1000_deepseek.csv"
MODEL       = "deepseek-r1:14b"
OLLAMA_URL  = "http://localhost:11434/api/generate"


def prepare_dataset():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    df = df[['question_id', 'question', 'answer']]

    if os.path.exists(OUTPUT_FILE):
        print(f"Found checkpoint at {OUTPUT_FILE}. Resuming...")
        df = pd.read_csv(OUTPUT_FILE)
        df['is_hallucinated'] = df['is_hallucinated'].fillna('')
    else:
        df['is_hallucinated'] = ''

    return df


def evaluate(df):
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if row.get('is_hallucinated') in ['Yes', 'No', 'yes', 'no']:
            continue

        prompt = f"""You are an expert evaluator for Bengali mathematical reasoning tasks.
Your task is to determine whether the given hallucinated_chain is hallucinated (i.e., incorrect or fabricated).

Question: {row['question']}
Reasoning Chain: {row['answer']}
Answer: {row['answer']}

Is this hallucinated_chain hallucinated? Respond ONLY with a JSON object like this:
{{"is_hallucinated": "Yes"}} or {{"is_hallucinated": "No"}}
Do not explain your reasoning or output anything else.
"""
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 8192,
                "num_predict": 2048,
                "temperature": 0,
            }
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=300)
            result = response.json()
            raw_response = result['response']

            cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

            if not cleaned.startswith("{"):
                m = re.search(r'\{.*\}', raw_response, flags=re.DOTALL)
                if m:
                    cleaned = m.group(0)

            try:
                is_hallucinated = json.loads(cleaned).get('is_hallucinated', '')
            except Exception:
                is_hallucinated = ''

            if is_hallucinated.strip().lower() not in ['yes', 'no']:
                print(f"Sample [{index}] - Malformed answer: '{is_hallucinated}'. Falling back...")
                is_hallucinated = 'Yes' if 'yes' in cleaned.lower() else 'No' if 'no' in cleaned.lower() else ''

            df.at[index, 'is_hallucinated'] = is_hallucinated
            print(f"Sample [{index}] - Question: {str(row['question'])[:50]}... | Result: {is_hallucinated}")
            df.to_csv(OUTPUT_FILE, index=False)

        except Exception as e:
            print(f"Error at index {index}: {e}")

    return df


def main():
    os.makedirs('Reasoning/Results', exist_ok=True)
    df = prepare_dataset()
    pending = (df['is_hallucinated'] == '').sum()
    print(f"Total: {len(df)} | Pending: {pending}")
    evaluate(df)
    print(f"\nDone! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
