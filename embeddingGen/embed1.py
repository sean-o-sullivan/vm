import os
import csv
import random
import stanza
import logging
from multiprocessing import Pool
from tqdm import tqdm
from embedding2 import get_embedding

SAMPLES_PER_AUTHOR = 100
SAMPLE_LENGTH = 1300  # Number of characters per sample
MIN_BOOK_LENGTH = SAMPLE_LENGTH * 2
MAX_BOOKS = 3  # Set this to your desired maximum number of books to process

nlp = stanza.Pipeline('en', processors='tokenize')

def get_text_sample(file_path, position):
    with open(file_path, 'r', encoding='utf-8') as file:
        file.seek(position)
        sample = file.read(SAMPLE_LENGTH)
    return sample

def process_sample(raw_sample):
    doc = nlp(raw_sample)
    sentences = [sent.text for sent in doc.sentences]
    
    # Drop the first and last sentences
    if len(sentences) > 2:
        processed_sample = ' '.join(sentences[1:-1])
    else:
        processed_sample = ''
    
    return processed_sample

def process_book(args):
    file_path, author, num_samples, output_file = args
    book_name = os.path.basename(file_path)
    logging.info(f"Processing book: {book_name} by {author}")

    file_size = os.path.getsize(file_path)
    
    embeddings = []
    for shard_id in range(num_samples):
        position = random.randint(0, max(0, file_size - SAMPLE_LENGTH))
        raw_sample = get_text_sample(file_path, position)
        processed_sample = process_sample(raw_sample)
        
        if len(processed_sample) < SAMPLE_LENGTH / 2:
            continue  
        
        embedding = get_embedding(processed_sample)
        row = [author, book_name, shard_id] + [embedding[key] for key in embedding]
        embeddings.append(row)
    
    logging.info(f"Saving {len(embeddings)} embeddings to {output_file}...")
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(embeddings)
    logging.info(f"Finished processing {file_path}.")
    return len(embeddings)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    input_dir = '/home/aiadmin/Desktop/datasets/bigText'
    output_file = 'output_embeddings1.csv'

    all_books = []

    # Collect all book paths
    for author_dir in os.listdir(input_dir):
        author_path = os.path.join(input_dir, author_dir)
        if os.path.isdir(author_path):
            for book_file in os.listdir(author_path):
                book_path = os.path.join(author_path, book_file)
                if os.path.isfile(book_path) and os.path.getsize(book_path) >= MIN_BOOK_LENGTH:
                    all_books.append((book_path, author_dir))

    # Shuffle the list of books
    random.shuffle(all_books)

    all_books = all_books[:MAX_BOOKS]

    args_list = [(book_path, author, SAMPLES_PER_AUTHOR, output_file) for book_path, author in all_books]

    embedding_keys = get_embedding("test").keys()
    headers = ['author', 'book', 'shardID'] + list(embedding_keys)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_book, args_list), total=len(args_list)))

    total_samples = sum(results)
    logging.info(f"Total samples processed: {total_samples}")

if __name__ == "__main__":
    main()
