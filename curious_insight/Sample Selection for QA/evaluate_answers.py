import pandas as pd
import json
import requests
import time
from tqdm import tqdm

def prepare_combined_dataset():
    print("Loading data...")
    ground_truth = pd.read_csv('Datasets/tydiqa_goldp_bengali.csv')
    deepseek_res = pd.read_csv('Sample Selection for QA/Results/deepseek_answers_bengali.csv')
    gemma_res = pd.read_csv('Sample Selection for QA/Results/gemma_answers_bengali.csv')
    qwen_res = pd.read_csv('Sample Selection for QA/Results/qwen_answers_bengali.csv')
    
    # Merge datasets
    deepseek_res = deepseek_res.rename(columns={'model_answer': 'deepseek_answer'})
    gemma_res = gemma_res.rename(columns={'model_answer': 'gemma_answer'})
    qwen_res = qwen_res.rename(columns={'model_answer': 'qwen_answer'})
    
    # Keep needed columns
    ground_truth = ground_truth[['id', 'question', 'answer_text']]
    deepseek_res = deepseek_res[['serial_id', 'deepseek_answer']]
    gemma_res = gemma_res[['serial_id', 'gemma_answer']]
    qwen_res = qwen_res[['serial_id', 'qwen_answer']]
    
    merged = ground_truth.merge(deepseek_res, left_on='id', right_on='serial_id', how='inner')
    merged = merged.merge(gemma_res, on='serial_id', how='inner')
    merged = merged.merge(qwen_res, on='serial_id', how='inner')
    
    merged = merged[['id', 'question', 'deepseek_answer', 'gemma_answer', 'qwen_answer', 'answer_text']]
    merged = merged.rename(columns={'answer_text': 'correct_answer'})
    
    # Add empty columns for scores
    merged['deepseek_score'] = 0
    merged['gemma_score'] = 0
    merged['qwen_score'] = 0
    
    return merged

def evaluate_with_qwen(df):
    url = "http://localhost:11434/api/generate"
    model_name = "qwen2.5:32b-instruct"
    
    print(f"Checking if {model_name} is available... Let's pull it if not.")
    
    # Evaluate each row
    for index, row in tqdm(df.iterrows(), total=len(df)):
        prompt = f"""You are an expert evaluator for Bengali Question Answering tasks.
Your task is to evaluate and compare three model answers against a correct ground truth answer.

Question: {row['question']}
Correct Answer: {row['correct_answer']}

Model Answers:
DeepSeek Answer: {row['deepseek_answer']}
Gemma Answer: {row['gemma_answer']}
Qwen Answer: {row['qwen_answer']}

Please score each model answer as 1 if it is correct, or 0 if it is incorrect.
Keep in mind these edge cases:
- Answers might have spelling mistakes or typos but are essentially correct.
- An answer may be provided in English instead of Bengali but conveys the exact same meaning.
- An answer might be partial but captures the core correct information.
If any of the above conditions are met and the core meaning matches the correct answer, score it as 1.

Respond ONLY with a JSON object containing the scores for each model, like this:
{{"deepseek_score": 1, "gemma_score": 0, "qwen_score": 1}}
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
            
            df.at[index, 'deepseek_score'] = scores.get('deepseek_score', 0)
            df.at[index, 'gemma_score'] = scores.get('gemma_score', 0)
            df.at[index, 'qwen_score'] = scores.get('qwen_score', 0)
        except Exception as e:
            print(f"Error at index {index}: {e}")
            
    return df

def main():
    combined_df = prepare_combined_dataset()
    print(f"Prepared dataset with {len(combined_df)} samples.")
    combined_df.to_csv('Sample Selection for QA/Results/combined_evaluation_unscored.csv', index=False)
    
    print("Evaluating answers using Qwen 2.5 32B Instruct...")
    evaluated_df = evaluate_with_qwen(combined_df)
    
    output_file = 'Sample Selection for QA/Results/combined_evaluation_scored.csv'
    evaluated_df.to_csv(output_file, index=False)
    print(f"Evaluation complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()
