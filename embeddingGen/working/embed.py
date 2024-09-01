import csv
import random
import pandas as pd
import logging
from tqdm import tqdm
from embedding2 import generateEmbedding

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv(input_file, output_file_70, output_file_30):
    # Read the CSV file and group texts by author
    author_texts = {}
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            author = row['author']
            if author not in author_texts:
                author_texts[author] = []
            author_texts[author].append(row)

    # Get the list of authors and shuffle it
    authors = list(author_texts.keys())
    random.shuffle(authors)

    # Calculate the split point
    split_point = int(len(authors) * 0.7)

    # Split authors into two groups
    authors_70 = authors[:split_point]
    authors_30 = authors[split_point:]

    # Write to output files
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




def process_entry(row, embedding_columns, expected_dim):
    author = row['author']
    book_name = row.get('book', '')
    sample_id = row.get('sample_id', '')
    processed_sample = row['cleaned_text']
    
    processed_sample = processed_sample.replace("#/#\\#|||#/#\\#|||#/#\\#", "")
    
    print(f"Processing sample_id: {sample_id}")
    print(f"Processed sample (first 100 chars): {processed_sample[:100]}")
    try:
        # Generate embedding
        embedding = generateEmbedding(processed_sample)
        
        # Check embedding dimension
        current_dim = len(embedding)
        if current_dim != expected_dim:
            raise ValueError(f"Inconsistent embedding dimension. Expected {expected_dim}, got {current_dim}")
        
        # Create a new row with the embedding data
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
    
    # Get the structure of the embedding dictionary using a sample entry
    sample_text = "This is a sample text to get the embedding structure."
    sample_embedding = generateEmbedding(sample_text)
    embedding_columns = list(sample_embedding.keys())
    expected_dim = len(sample_embedding)
    
    print(f"Expected embedding dimension: {expected_dim}")
    
    # Process each entry and generate embeddings
    tqdm.pandas(desc="Processing entries")
    result_df = df.progress_apply(lambda row: process_entry(row, embedding_columns, expected_dim), axis=1)
    
    # Remove any rows that returned None due to errors
    result_df = result_df.dropna()
    
    # Final check of embedding dimensions
    embedding_cols = [col for col in result_df.columns if col not in ['author', 'book', 'sample_id']]
    actual_dim = len(embedding_cols)
    if actual_dim != expected_dim:
        logging.error(f"Final embedding dimension mismatch. Expected {expected_dim}, got {actual_dim}")
    else:
        logging.info(f"All embeddings have the expected dimension of {expected_dim}")
    
    # Save the result to the output CSV
    result_df.to_csv(output_file, index=False)
    logging.info(f"Processing completed. Embeddings saved to {output_file}")

def main():
    # Process AGG.csv
    process_csv('AGG.csv', 'AGG_70.csv', 'AGG_30.csv')
    generate_embeddings('AGG_70.csv', 'AGG_70_embeddings.csv')
    generate_embeddings('AGG_30.csv', 'AGG_30_embeddings.csv')

    # Process ABB.csv
    process_csv('ABB.csv', 'ABB_70.csv', 'ABB_30.csv')
    generate_embeddings('ABB_70.csv', 'ABB_70_embeddings.csv')
    generate_embeddings('ABB_30.csv', 'ABB_30_embeddings.csv')

    print("Processing complete. All output files have been created.")

if __name__ == "__main__":
    main()
