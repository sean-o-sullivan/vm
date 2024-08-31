import pandas as pd
import logging
import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def process_entry(row):
    author = row['author']
    book_name = row['book']
    sample_id = row['sample_id']
    processed_sample = row['processed_sample']
    # Remove the custom delimiters from the processed_sample
    processed_sample = processed_sample.replace("#/#\\#|||#/#\\#|||#/#\\#", "")
    
    print(f"Processing sample_id: {sample_id}")
    print(f"Author: {author}")
    print(f"Book: {book_name}")
    print(f"Processed sample (first 100 chars): {processed_sample[:100]}")
    
    input("Press Enter to continue...")  # Wait for my user input 

def main():
    input_csv = '/home/aiadmin/Desktop/code/vm/embeddingGen/Thursday/results_10KSample.csv'
    df = pd.read_csv(input_csv)
    
    if df.empty:
        logging.error("No entries found in results.csv. Exiting.")
        return

    print(f"CSV Headers: {df.columns.tolist()}")
    print(f"Total entries: {len(df)}")
    tqdm.pandas(desc="Processing entries")
    df.progress_apply(lambda row: process_entry(row), axis=1)

    logging.info("Processing completed.")

if __name__ == "__main__":
    main()
