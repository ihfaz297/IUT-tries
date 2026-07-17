import pandas as pd
from bert_score import score
import os

def main():
    dataset_path = "Sample Selection for Summ/Datasets/CHQSumm.csv"
    deepseek_path = "Sample Selection for Summ/Results/deepseek_summaries_CHQSumm.csv"
    qwen_path = "Sample Selection for Summ/Results/qwen_summaries_CHQSumm.csv"
    gemma_path = "Sample Selection for Summ/Results/gemma_summaries_CHQSumm.csv"
    output_path = "Sample Selection for Summ/Results/combined_summaries_bertscore.csv"

    def load_generated_summary(path, col_name):
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Make sure we only grab id and the generated summary
            if 'generated_summary' in df.columns:
                return df[['id', 'generated_summary']].rename(columns={'generated_summary': col_name})
        return pd.DataFrame(columns=['id', col_name])

    print("Loading datasets...")
    # Load base dataset
    df = pd.read_csv(dataset_path)
    df = df[['id', 'question', 'summary']]

    # Load models
    df_deepseek = load_generated_summary(deepseek_path, 'deepseek_summary')
    df_qwen = load_generated_summary(qwen_path, 'qwen_summary')
    df_gemma = load_generated_summary(gemma_path, 'gemma_summary')

    # Merge
    df = df.merge(df_deepseek, on='id', how='left')
    df = df.merge(df_qwen, on='id', how='left')
    df = df.merge(df_gemma, on='id', how='left')

    # Fill NaN with a space to prevent bert_score from crashing on empty strings
    df['summary'] = df['summary'].fillna(" ")
    df['deepseek_summary'] = df['deepseek_summary'].fillna(" ")
    df['qwen_summary'] = df['qwen_summary'].fillna(" ")
    df['gemma_summary'] = df['gemma_summary'].fillna(" ")

    refs = df['summary'].tolist()

    def get_bert_f1(preds, refs):
        # Calculate BERT score for Bengali (lang="bn")
        # Returns Precision, Recall, F1. We will extract F1.
        P, R, F1 = score(preds, refs, lang="bn", verbose=True)
        return F1.numpy()

    print("Calculating BERT scores. This may take a bit as it downloads/loads the multilingual BERT model...")

    if df['deepseek_summary'].str.strip().any():
        print("Calculating DeepSeek BERTScore...")
        df['deepseek_bertscore_f1'] = get_bert_f1(df['deepseek_summary'].tolist(), refs)
    else:
        df['deepseek_bertscore_f1'] = None

    if df['qwen_summary'].str.strip().any():
        print("Calculating Qwen BERTScore...")
        df['qwen_bertscore_f1'] = get_bert_f1(df['qwen_summary'].tolist(), refs)
    else:
        df['qwen_bertscore_f1'] = None

    if df['gemma_summary'].str.strip().any():
        print("Calculating Gemma BERTScore...")
        df['gemma_bertscore_f1'] = get_bert_f1(df['gemma_summary'].tolist(), refs)
    else:
        df['gemma_bertscore_f1'] = None

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Done! Combined file saved to {output_path}")

if __name__ == "__main__":
    main()
