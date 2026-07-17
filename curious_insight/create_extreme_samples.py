import pandas as pd

def create_combined_sample(sorted_file, output_file):
    # Load the sorted dataset
    print(f"Reading {sorted_file}...")
    df = pd.read_csv(sorted_file)

    # Get the top 500 (highest length)
    top_500 = df.head(500)
    
    # Get the bottom 500 (lowest length)
    bottom_500 = df.tail(500)

    # Combine them
    combined_df = pd.concat([top_500, bottom_500], ignore_index=True)

    # Save to new CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined dataset (1000 rows) to {output_file}")
    print(f"Top 500 length range: {top_500['reasoning_len'].max()} to {top_500['reasoning_len'].min()}")
    print(f"Bottom 500 length range: {bottom_500['reasoning_len'].max()} to {bottom_500['reasoning_len'].min()}")

if __name__ == "__main__":
    sorted_input = "/home/bio/Desktop/Thesis-401/reasoning based/BanglaHalluEval/SOMADHAN_SORTED.csv"
    output_sample = "/home/bio/Desktop/Thesis-401/reasoning based/BanglaHalluEval/SOMADHAN_EXTREMES_1000.csv"
    create_combined_sample(sorted_input, output_sample)
