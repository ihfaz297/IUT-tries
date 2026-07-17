import pandas as pd
import requests
import json
import tqdm
import os
import time
import re

def calculate_average_summary_length(csv_path):
    # Read the dataset
    df = pd.read_csv(csv_path)
    # Calculate lengths of summaries
    lengths = df['summary'].dropna().apply(lambda x: len(str(x).split()))
    return int(lengths.mean())

def generate_summaries(csv_path, output_path, model_name, avg_length):
    df = pd.read_csv(csv_path)
    
    # Check if we already have some generated summaries
    generated_summaries = []
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        generated_summaries = existing_df['generated_summary'].tolist()
        df = df.iloc[len(generated_summaries):]
    
    if len(df) == 0:
        print("All summaries already generated")
        return

    print(f"Generating summaries for {len(df)} questions using {model_name}...")
    
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        q_id = row['id']
        question = row['question']
        
        prompt = (f"You are a helpful assistant. Summarize the following Bengali medical text in Bengali. The summary should be approximately "
                  f"{avg_length} words long. Provide ONLY the summary in Bengali and nothing else.\n\nText:\n{question}")
        
        try:
            response = requests.post('http://localhost:11434/api/generate', json={
                'model': model_name,
                'prompt': prompt,
                'stream': False
            })
            if response.status_code == 200:
                answer = response.json()['response'].strip()
                # Remove deepseek thinking process if present
                answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
                generated_summaries.append(answer)
            else:
                print(f"Error for ID {q_id}: {response.status_code}")
                generated_summaries.append("")
        except Exception as e:
            print(f"Exception for ID {q_id}: {e}")
            generated_summaries.append("")
            
        # Optional: rate limiting
        time.sleep(0.1)
        
        # Save incrementally
        temp_df = pd.read_csv(csv_path)
        temp_df = temp_df.iloc[:len(generated_summaries)]
        temp_df['generated_summary'] = generated_summaries
        temp_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    csv_file = "Sample Selection for Summ/Datasets/CHQSumm.csv"
    
    # Using DeepSeek-R1 14b for generation
    model = "deepseek-r1:14b"
    output_file = "Sample Selection for Summ/Results/deepseek_summaries_CHQSumm.csv"
    
    # Create Results dir if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("Calculating average summary length...")
    avg_len = calculate_average_summary_length(csv_file)
    print(f"Average summary length: {avg_len} words")
    
    print(f"Starting summary generation with {model}...")
    generate_summaries(csv_file, output_file, model, avg_len)

