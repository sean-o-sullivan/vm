import pandas as pd
import random

def main():
    input_csv = '/home/aiadmin/Desktop/code/vm/embeddingGen/Thursday/results_10KSample.csv'
    output_csv = 'selected_samples.csv'

    
    df = pd.read_csv(input_csv)
    
    if df.empty:
        print("No entries found in the CSV. Exiting.")
        return

    author_sample_counts = df['author'].value_counts()
    eligible_authors = author_sample_counts[author_sample_counts >= 5].index.tolist()
    
    print(f"Total number of eligible authors with at least 5 samples: {len(eligible_authors)}")
    selected_authors = random.sample(eligible_authors, min(1000, len(eligible_authors)))
    print(f"Number of selected authors: {len(selected_authors)}")
    selected_rows = []

    for author in selected_authors:
        
        author_samples = df[df['author'] == author]
        selected_samples = author_samples.sample(n=5)
        selected_rows.append(selected_samples)

    
    result_df = pd.concat(selected_rows, ignore_index=True)
    result_df.to_csv(output_csv, index=False, quoting=1)

    print(f"Selected samples saved to {output_csv}")

if __name__ == "__main__":
    main()
