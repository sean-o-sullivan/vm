import csv
import logging
from tqdm import tqdm
from embedding2 import generateEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_entry(entry, fieldnames, output_file):
    author = entry['author']
    book_name = entry['book']
    sample_id = entry['sample_id']
    processed_sample = entry['processed_sample']
    
    # Remove the custom delimiters from the processed_sample
    processed_sample = processed_sample.replace("#/#\\#|||#/#\\#|||#/#\\#", "")
    
    print(f"Processing sample_id: {sample_id}")
    print(f"Processed sample (first 100 chars): {processed_sample[:100]}")

    try:
        
        embedding = generateEmbedding(processed_sample)

        
        row = {
            'author': author,
            'book': book_name,
            'sample_id': sample_id
        }
        row.update(embedding)  

        
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
    except Exception as e:
        logging.error(f"Error processing sample_id {sample_id}: {str(e)}")

def main():
    input_csv = '/home/aiadmin/Desktop/code/vm/embeddingGen/Thursday/results_10KSample.csv'
    output_file = 'output_embeddings.csv'

    
    with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        entries = list(reader)
    
    if not entries:
        logging.error("No entries found in results.csv. Exiting.")
        return

    print(f"CSV Headers: {reader.fieldnames}")
    print(f"Total entries: {len(entries)}")

    
    sample_text = "This is a sample text to generate embedding structure."
    sample_embedding = generateEmbedding(sample_text)
    fieldnames = ['author', 'book', 'sample_id'] + list(sample_embedding.keys())

    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    
    for entry in tqdm(entries, desc="Processing entries"):
        process_entry(entry, fieldnames, output_file)

    logging.info(f"Processing completed. Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()
