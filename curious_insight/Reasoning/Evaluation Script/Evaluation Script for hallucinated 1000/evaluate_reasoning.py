import pandas as pd
import json
import requests
import time
import os
from tqdm import tqdm

def prepare_dataset():
    print("Loading data...")
    df = pd.read_csv('Reasoning/1000_hallucinated Samples/somadhan_1000_hallucinated.csv')
    df = df[['id', 'question', 'hallucinated_chain', 'hallucinated_answer']]
    
    # Load checkpoint if it exists
    checkpoint_file = 'Reasoning/Results/reasoning_evaluation_scored.csv'
    if os.path.exists(checkpoint_file):
        print(f"Found existing checkpoint at {checkpoint_file}. Resuming from there...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        # Assuming the check point has 'is_hallucinated', initialize our return df with it
        df = checkpoint_df
        # Replace NaNs with empty strings for consistent checking
        df['is_hallucinated'] = df['is_hallucinated'].fillna('')
    else:
        df['is_hallucinated'] = ''
        
    return df, checkpoint_file

def evaluate_with_qwen(df, checkpoint_file):
    url = "http://localhost:11434/api/generate"
    model_name = "qwen2.5:32b-instruct"

    print(f"Checking if {model_name} is available... Let's pull it if not.")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Skip if already evaluated
        if row.get('is_hallucinated') in ['Yes', 'No', 'yes', 'no']:
            continue
            
        prompt = f"""You are an expert evaluator for Bengali mathematical reasoning tasks.
Your task is to determine whether the given hallucinated_chain is hallucinated (i.e., incorrect or fabricated).

Question: {row['question']}
Reasoning Chain: {row['hallucinated_chain']}
Answer: {row['hallucinated_answer']}

Is this hallucinated_chain hallucinated? Respond ONLY with a JSON object like this:
{{"is_hallucinated": "Yes"}} or {{"is_hallucinated": "No"}}
Do not explain your reasoning or output anything else.
"""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(url, json=payload)
            result = response.json()
            scores = json.loads(result['response'])

            df.at[index, 'is_hallucinated'] = scores.get('is_hallucinated', '')
            
            # Save checkpoint after each iteration
            df.to_csv(checkpoint_file, index=False)
            
        except Exception as e:
            print(f"Error at index {index}: {e}")

    return df

def main():
    os.makedirs('Reasoning/Results', exist_ok=True)
    df, checkpoint_file = prepare_dataset()
    print(f"Prepared dataset with {len(df)} samples.")

    print("Evaluating reasoning samples using Qwen 2.5 32B Instruct...")
    evaluated_df = evaluate_with_qwen(df, checkpoint_file)

    print(f"Evaluation complete! Results saved to {checkpoint_file}")

if __name__ == "__main__":
    main()
