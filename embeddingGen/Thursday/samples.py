import os
import csv
import random
import stanza
import logging
import json
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict

SAMPLES_PER_AUTHOR = 100
SAMPLE_LENGTH = 1300  # Number of characters per sample
MIN_BOOK_LENGTH = SAMPLE_LENGTH * 2
NO_TOUCH_ZONE = 1000  # First 1000 characters will be skipped
MAX_BOOKS = 3  # Maximum number of books to process in total

nlp = stanza.Pipeline('en', processors='tokenize')

def get_text_sample(file_path, position):
    with open(file_path, 'r', encoding='utf-8') as file:
        file.seek(position)
        sample = file.read(SAMPLE_LENGTH)
    return sample

def process_sample(raw_sample):
    doc = nlp(raw_sample)
    sentences = [sent.text for sent in doc.sentences]
    
    if len(sentences) > 2:
        processed_sample = ' '.join(sentences[1:-1])
    else:
        processed_sample = ''
    
    return processed_sample

def save_samples_to_csv(author, book_name, samples, sample_file):
    with open(sample_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['author', 'book', 'sample_id', 'raw_sample', 'processed_sample']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for i, (raw_sample, processed_sample) in enumerate(samples):
            writer.writerow({
                'author': author,
                'book': book_name,
                'sample_id': i,
                'raw_sample': raw_sample,
                'processed_sample': processed_sample
            })

def process_book(args):
    file_path, author, max_samples, sample_file = args
    book_name = os.path.basename(file_path)
    logging.info(f"Processing book: {book_name} by {author}")

    file_size = os.path.getsize(file_path)
    effective_file_size = max(0, file_size - NO_TOUCH_ZONE)
    max_possible_samples = max(1, (effective_file_size - SAMPLE_LENGTH) // (SAMPLE_LENGTH // 2))
    num_samples = min(max_samples, max_possible_samples)
    
    samples = []
    for shard_id in range(num_samples):
        position = random.randint(NO_TOUCH_ZONE, max(NO_TOUCH_ZONE, file_size - SAMPLE_LENGTH))
        raw_sample = get_text_sample(file_path, position)
        processed_sample = process_sample(raw_sample)
        
        if len(processed_sample) < SAMPLE_LENGTH / 2:
            continue  
        
        samples.append((raw_sample, processed_sample))
    
    save_samples_to_csv(author, book_name, samples, sample_file)
    
    logging.info(f"Finished processing {file_path}.")
    return len(samples)

def parse_custom_id(custom_id):
    # Extract author and file parts from the custom_id
    author_part, file_part = custom_id.split('-', 1)
    author_name = author_part.replace('_', '__')
    file_name = file_part.split('_', 1)[-1]  # Extract filename part
    return author_name, file_name

def load_eligible_books(jsonl_path):
    eligible_books = defaultdict(set)
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            custom_id = entry['custom_id']
            response_content = entry['response']['body']['choices'][0]['message']['content']
            if response_content == "YES":
                author_name, file_name = parse_custom_id(custom_id)
                eligible_books[author_name].add(file_name)
    return eligible_books

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    input_dir = '/home/aiadmin/Desktop/datasets/bigText'
    sample_file = 'samples.csv'
    jsonl_path = 'embeddingGen/batch_dataset_classification_output.jsonl'

    # Load the set of eligible books
    eligible_books = load_eligible_books(jsonl_path)

    # Create a list to store all eligible book paths
    all_books = []

    # Collect all eligible book paths
    for author_dir in os.listdir(input_dir):
        author_path = os.path.join(input_dir, author_dir)
        if os.path.isdir(author_path):
            if author_dir in eligible_books:
                for book_file in os.listdir(author_path):
                    book_path = os.path.join(author_path, book_file)
                    if os.path.isfile(book_path) and os.path.getsize(book_path) >= MIN_BOOK_LENGTH + NO_TOUCH_ZONE:
                        if book_file in eligible_books[author_dir]:
                            all_books.append((book_path, author_dir))

    # Shuffle and limit to MAX_BOOKS
    random.shuffle(all_books)
    selected_books = all_books[:MAX_BOOKS]

    args_list = []
    for book_path, author in selected_books:
        args_list.append((book_path, author, SAMPLES_PER_AUTHOR, sample_file))

    with open(sample_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['author', 'book', 'sample_id', 'raw_sample', 'processed_sample']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_book, args_list), total=len(args_list)))

    total_samples = sum(results)
    logging.info(f"Total samples processed: {total_samples}")

if __name__ == "__main__":
    main()
