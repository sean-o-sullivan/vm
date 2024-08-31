import pandas as pd
import logging
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    input_csv = '/home/aiadmin/Desktop/code/vm/embeddingGen/Thursday/results_10KSample.csv'
    output_file = 'output_raw_texts_with_delimiters.csv'
    df = pd.read_csv(input_csv)
    
    if df.empty:
        logging.error("No entries found in results.csv. Exiting.")
        return

    print(f"all of the CSV Headers: {df.columns.tolist()}")
    print(f"Total entries: {len(df)}")
    unique_authors = df['author'].nunique()
    print(f"Unique authors in the dataset: {unique_authors}")
    
    
    selected_authors = random.sample(df['author'].unique().tolist(), 1000)
    
    
    with open(output_file, 'w') as f:
        
        f.write("author,book,sample_id,processed_sample\n")
        
        
        for author in tqdm(selected_authors, desc="Processing authors"):
            author_df = df[df['author'] == author]
            
            if len(author_df) >= 5:
                selected_texts = author_df.sample(5)
            else:
                selected_texts = author_df
            
            for _, row in selected_texts.iterrows():
                processed_sample = row['processed_sample']  # Keep my delimiter
                line = f"{row['author']},{row['book']},{row['sample_id']},\"{processed_sample}\"\n"
                f.write(line)

    logging.info(f"Processing completed. Raw texts with delimiters saved to {output_file}")

if __name__ == "__main__":
    main()
