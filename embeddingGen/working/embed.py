import csv
import random
import pandas as pd
import logging
from tqdm import tqdm
from embedding2 import generateEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv(input_file, output_file_70, output_file_30):
    author_texts = {}
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            author = row['author']
            if author not in author_texts:
                author_texts[author] = []
            author_texts[author].append(row)

    authors = list(author_texts.keys())
    random.shuffle(authors)
    split_point = int(len(authors) * 0.7)
    authors_70 = authors[:split_point]
    authors_30 = authors[split_point:]
    write_output(output_file_70, author_texts, authors_70)
    write_output(output_file_30, author_texts, authors_30)

def write_output(output_file, author_texts, authors):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(next(iter(author_texts.values()))[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for author in authors:
            for row in author_texts[author]:
                writer.writerow(row)

def process_entry(row, expected_keys, key_counts):
    author = row['author']
    book_name = row.get('book', '')
    sample_id = row.get('sample_id', '')
    processed_sample = row['cleaned_text']
    
    processed_sample = processed_sample.replace("#/#\\#|||#/#\\#|||#/#\\#", "")
    
    print(f"Processing sample_id: {sample_id}")
    print(f"Processed sample (first 100 chars): {processed_sample[:100]}")
    try:
        embedding = generateEmbedding(processed_sample)
        current_keys = set(embedding.keys())
        new_keys = current_keys - expected_keys
        missing_keys = expected_keys - current_keys

        if new_keys:
            print(f"New keys found for sample_id {sample_id}: {new_keys}")
        if missing_keys:
            print(f"Missing keys for sample_id {sample_id}: {missing_keys}")

        for key in current_keys:
            key_counts[key] += 1
            new_row = {
            'author': author,
            'book': book_name,
            'sample_id': sample_id
        }
        new_row.update(embedding)
        return pd.Series(new_row)
    except Exception as e:
        logging.error(f"Error processing sample_id {sample_id}: {str(e)}")
        return None

def generate_embeddings(input_file, output_file):
    df = pd.read_csv(input_file)
    
    if df.empty:
        logging.error(f"No entries found in {input_file}. Skipping.")
        return
    
    print(f"CSV Headers: {df.columns.tolist()}")
    print(f"Total entries: {len(df)}")
    sample_text = "This is a sample text to get the embedding structure."
    sample_embedding = generateEmbedding(sample_text)
    expected_keys = set(sample_embedding.keys())
    print(f"Expected embedding keys: {expected_keys}")
    print(f"Expected embedding dimension: {len(expected_keys)}")
    key_counts = defaultdict(int)
    tqdm.pandas(desc="Processing entries")
    result_df = df.progress_apply(lambda row: process_entry(row, expected_keys, key_counts), axis=1)
    result_df = result_df.dropna()
    embedding_cols = [col for col in result_df.columns if col not in ['author', 'book', 'sample_id']]
    actual_keys = set(embedding_cols)
    
    print("\nEmbedding Key Analysis:")
    print(f"Expected keys: {len(expected_keys)}")
    print(f"Actual keys: {len(actual_keys)}")
    
    new_keys = actual_keys - expected_keys
    missing_keys = expected_keys - actual_keys
    
    if new_keys:
        print(f"New keys found: {new_keys}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    
    print("\nKey Frequency Analysis:")
    total_samples = len(result_df)
    for key, count in key_counts.items():
        frequency = count / total_samples
        print(f"{key}: {frequency:.2%}")
    result_df.to_csv(output_file, index=False)
    logging.info(f"Processing completed. Embeddings saved to {output_file}")

def main():
    process_csv('AGG.csv', 'AGG_70.csv', 'AGG_30.csv')
    generate_embeddings('AGG_70.csv', 'AGG_70_embeddings.csv')
    generate_embeddings('AGG_30.csv', 'AGG_30_embeddings.csv'
    process_csv('ABB.csv', 'ABB_70.csv', 'ABB_30.csv')
    generate_embeddings('ABB_70.csv', 'ABB_70_embeddings.csv')
    generate_embeddings('ABB_30.csv', 'ABB_30_embeddings.csv')

    print("Processing complete. All output files have been created.")

if __name__ == "__main__":
    main()

