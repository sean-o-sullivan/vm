import pandas as pd
import logging
from tqdm import tqdm
from embedding2 import generateEmbedding
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_entry(row, embedding_columns):
    author = row['author']
    text = row['text']
    try:
        embedding = generateEmbedding(text)

        new_row = {
            'author': author,
            'book': text[:100],
        }
        new_row.update(embedding) 
        return pd.Series(new_row)
    except Exception as e:
        logging.error(f"Error processing the text (first 100 chars): {text[:100]}: {str(e)}")
        return None

def main():
    input_csv = '/home/aiadmin/Downloads/cleaned_reuters_corpus.csv'
    output_file = '/home/aiadmin/Downloads/output_2embeddings_Reuters.csv'

    df = pd.read_csv(input_csv)
    if df.empty:
        logging.error("No entries found in input CSV. Exiting.")
        return
    
    sample_text = "Sample text for generating embedding structure."
    sample_embedding = generateEmbedding(sample_text)
    embedding_columns = list(sample_embedding.keys())
    # result_df = pd.DataFrame(columns=['author', 'book'] + embedding_columns)
    file_exists = os.path.isfile(output_file)
    result_df = pd.DataFrame(columns=['author', 'book'] + embedding_columns)

    if not file_exists:
        result_df.to_csv(output_file, index=False)
        logging.info(f"Created new output file: {output_file}")

    processed_entries = []
    save_interval = 10 

    tqdm.pandas(desc="Processing entries")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        processed_row = process_entry(row, embedding_columns)
        if processed_row is not None:
            processed_entries.append(processed_row)

        if len(processed_entries) >= save_interval:
            temp_df = pd.DataFrame(processed_entries)
            temp_df.to_csv(output_file, mode='a', header=False, index=False)
            processed_entries = [] 

    if processed_entries:
        temp_df = pd.DataFrame(processed_entries)
        temp_df.to_csv(output_file, mode='a', header=False, index=False)

    logging.info(f"Processing is completed. Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()