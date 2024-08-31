import pandas as pd
import logging
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s - %(message)s')

# a simple, rare character as the delimiter, is it though?
DELIMITER = '|'

def generate_and_verify_csv(input_csv, output_file):
    df = pd.read_csv(input_csv)
    
    if df.empty:
        logging.error("No entries found in results.csv. Exiting.")
        return

    print(f"all of the CSV Headers: {df.columns.tolist()}")
    print(f"Total entries: {len(df)}")
    unique_authors = df['author'].nunique()
    print(f"Unique authors in the dataset: {unique_authors}")
    
    
    selected_authors = random.sample(df['author'].unique().tolist(), 1000)
    
    # Prepare to write output CSV with the new delimiter
    with open(output_file, 'w') as f:
        f.write(f"author{DELIMITER}book{DELIMITER}sample_id{DELIMITER}processed_sample\n")
        
        
        for author in tqdm(selected_authors, desc="Processing authors"):
            author_df = df[df['author'] == author]
            
            if len(author_df) >= 5:
                selected_texts = author_df.sample(5)
            else:
                selected_texts = author_df
            
            for _, row in selected_texts.iterrows():
                processed_sample = row['processed_sample']  # Keep the text as is
                
                line = f"{row['author']}{DELIMITER}{row['book']}{DELIMITER}{row['sample_id']}{DELIMITER}{processed_sample}\n"
                f.write(line)

    logging.info(f"CSV generation completed. File saved as {output_file}")

    verify_csv(output_file)

def verify_csv(output_file):
    try:
        df = pd.read_csv(output_file, delimiter=DELIMITER)
        print(f"CSV successfully loaded with {len(df)} rows and the following columns: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error loading CSV: {str(e)}")

if __name__ == "__main__":
    input_csv = '/home/aiadmin/Desktop/code/vm/embeddingGen/Thursday/results_10KSample.csv'
    output_file = 'output_raw_texts_with_pipe_delimiter.csv'
    generate_and_verify_csv(input_csv, output_file)
