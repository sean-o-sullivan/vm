import csv
import pandas as pd
import logging
from tqdm import tqdm
from embedding2 import generateEmbedding
from collections import defaultdict
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_entry(row, expected_keys, key_counts, embedding_id, text_column):
    author = row['author']
    processed_sample = row[text_column]
    
    processed_sample = processed_sample.replace("#/#\\#|||#/#\\#|||#/#\\#", "")
    processed_sample = processed_sample.replace("'''", "")
    
    print(f"Processing embedding_id: {embedding_id}")
    print(f"Processed sample (first 100 chars): {processed_sample[:100]}")
    try:
        #embedding
        embedding = generateEmbedding(processed_sample)
        current_keys = set(embedding.keys())
        new_keys = current_keys - expected_keys
        missing_keys = expected_keys - current_keys

        if new_keys:
            print(f"New keys found for embedding_id {embedding_id}: {new_keys}")
        if missing_keys:
            print(f"Missing keys for embedding_id {embedding_id}: {missing_keys}")
        for key in current_keys:
            key_counts[key] += 1
        new_row = {
            'embedding_id': embedding_id,
            'author': author,
            f'{text_column}_embedding': str(list(embedding.values()))
        }
        new_row.update(row)
        return pd.Series(new_row)
    except Exception as e:
        logging.error(f"Error processing embedding_id {embedding_id}: {str(e)}")
        return None

def generate_embeddings(input_file, output_file):
    df = pd.read_csv(input_file)
    
    if df.empty:
        logging.error(f"No entries found in {input_file}. Skipping.")
        return
    print(f"CSV Headers: {df.columns.tolist()}")
    print(f"Total entries: {len(df)}")
    
    if 'generated_text' in df.columns:
        text_column = 'generated_text'
    elif 'generated_mimicry' in df.columns:
        text_column = 'generated_mimicry'
    else:
        logging.error(f"No valid text column found in {input_file}. Skipping.")
        return
    
    sample_text = df[text_column].iloc[0]
    sample_embedding = generateEmbedding(sample_text)
    expected_keys = set(sample_embedding.keys())
    print(f"Expected embedding keys: {expected_keys}")
    print(f"Expected embedding dimension: {len(expected_keys)}")
    
    key_counts = defaultdict(int)
    
    processed_rows = []
    embedding_id = 1 
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing entries"):
        processed_row = process_entry(row, expected_keys, key_counts, embedding_id, text_column)
        if processed_row is not None:
            processed_rows.append(processed_row)
            embedding_id += 1  
        if embedding_id % 20 == 0:
            temp_df = pd.DataFrame(processed_rows)
            temp_df.to_csv(output_file, index=False, mode='a', header=not pd.io.common.file_exists(output_file))
            processed_rows = [] 
    if processed_rows:
        temp_df = pd.DataFrame(processed_rows)
        temp_df.to_csv(output_file, index=False, mode='a', header=not pd.io.common.file_exists(output_file))
    
    logging.info(f"Processing completed. Embeddings saved to {output_file}")

def main():
    file_pairs = [
        ('mimicry_samples_GPT3ABB_30.csv', 'mimicry_samples_GPT3ABB_30_embeddings.csv'),
        ('mimicry_samples_GPT3AGG_30.csv', 'mimicry_samples_GPT3AGG_30_embeddings.csv'),
        ('mimicry_samples_GPT4oABB_30.csv', 'mimicry_samples_GPT4oABB_30_embeddings.csv'),
        ('mimicry_samples_GPT4oAGG_30.csv', 'mimicry_samples_GPT4oAGG_30_embeddings.csv'),
        ('mimicry_samples_GPT4TABB_30.csv', 'mimicry_samples_GPT4TABB_30_embeddings.csv'),
        ('mimicry_samples_GPT4TAGG_30.csv', 'mimicry_samples_GPT4TAGG_30_embeddings.csv'),
        ('topic_based_samples_GPT3ABB_30.csv', 'topic_based_samples_GPT3ABB_30_embeddings.csv'),
        ('topic_based_samples_GPT3AGG_30.csv', 'topic_based_samples_GPT3AGG_30_embeddings.csv'),
        ('topic_based_samples_GPT4oABB_30.csv', 'topic_based_samples_GPT4oABB_30_embeddings.csv'),
        ('topic_based_samples_GPT4oAGG_30.csv', 'topic_based_samples_GPT4oAGG_30_embeddings.csv'),
        ('topic_based_samples_GPT4TABB_30.csv', 'topic_based_samples_GPT4TABB_30_embeddings.csv'),
        ('topic_based_samples_GPT4TAGG_30.csv', 'topic_based_samples_GPT4TAGG_30_embeddings.csv'),
    ]

    for input_file, output_file in file_pairs:
        print(f"Processing {input_file}...")
        generate_embeddings(input_file, output_file)

    print("Processing complete. All output files have been created.")

if __name__ == "__main__":
    main()
