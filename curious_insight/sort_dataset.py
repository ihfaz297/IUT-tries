import pandas as pd
import re

def process_and_sort_dataset(file_path, output_path):
    # Load the CSV
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)

    # Function to get the reasoning part and its length
    def get_reasoning_length(text):
        if pd.isna(text):
            return 0
        # Remove the '#### [Value]' part at the end
        # This regex looks for #### followed by any characters until the end of the string
        reasoning_only = re.sub(r'####\s*.*$', '', str(text), flags=re.DOTALL).strip()
        return len(reasoning_only)

    # Add a temporary column for sorting
    df['reasoning_len'] = df['answer'].apply(get_reasoning_length)

    # Sort descending based on length
    print("Sorting dataset by reasoning length...")
    df_sorted = df.sort_values(by='reasoning_len', ascending=False)

    # Remove the temporary length column if you want the original structure
    # Or keep it to verify. I'll keep it for now as per instructions "calculate the length".
    
    # Save to new CSV
    df_sorted.to_csv(output_path, index=False)
    print(f"Saved sorted dataset to {output_path}")
    
    # Show top 5 longest reasonings
    print("\nTop 5 Longest Reasonings (Previews):")
    for idx, row in df_sorted.head(5).iterrows():
        print(f"\nLength: {row['reasoning_len']}")
        print(f"Reasoning Preview: {row['answer'][:200]}...")

if __name__ == "__main__":
    input_file = "/home/bio/Desktop/Thesis-401/reasoning based/BanglaHalluEval/SOMADHAN.csv"
    output_file = "/home/bio/Desktop/Thesis-401/reasoning based/BanglaHalluEval/SOMADHAN_SORTED.csv"
    process_and_sort_dataset(input_file, output_file)
