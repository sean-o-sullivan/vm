import os
import csv
import random
import stanza
import logging
import json
from tqdm import tqdm
from collections import defaultdict

SAMPLES_PER_AUTHOR = 100
SAMPLE_LENGTH = 1300  # Number of characters per sample
MIN_BOOK_LENGTH = SAMPLE_LENGTH * 2
NO_TOUCH_ZONE = 1000  # the First 1000 characters will be skipped
MAX_BOOKS = 3  # Maximum number of books to process in total

BIG_TEXT_DIR = '/home/aiadmin/Desktop/datasets/bigText'
JSONL_PATH = '/home/aiadmin/Desktop/code/vm/embeddingGen/batch_dataset_classification_output.jsonl'

try:
    nlp = stanza.Pipeline('en', processors='tokenize')
except Exception as e:
    logging.error(f"Failed to initialize Stanza pipeline: {e}")
    raise

def get_text_sample(file_path, position):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file.seek(position)
            sample = file.read(SAMPLE_LENGTH)
        return sample
    except Exception as e:
        logging.error(f"Error reading text sample from {file_path} at position {position}: {e}")
        return None

def process_sample(raw_sample):
    try:
        doc = nlp(raw_sample)
        sentences = [sent.text for sent in doc.sentences]
        
        if (len(sentences) > 2):
            processed_sample = ' '.join(sentences[1:-1])
        else:
            processed_sample = ''
        
        return processed_sample
    except Exception as e:
        logging.error(f"Error processing sample: {e}")
        return None

def save_samples_to_csv(author, book_name, samples, sample_file):
    try:
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
    except Exception as e:
        logging.error(f"Error saving samples to CSV for book {book_name} by {author}: {e}")

def process_book(args):
    file_path, author, max_samples, sample_file = args
    book_name = os.path.basename(file_path)
    logging.info(f"Processing book: {book_name} by {author}")

    try:
        file_size = os.path.getsize(file_path)
    except Exception as e:
        logging.error(f"Failed to get size for  {file_path}: {e}")
        return 0

    if file_size < MIN_BOOK_LENGTH + NO_TOUCH_ZONE:
        logging.warning(f"File {book_name} is too short .")
        return 0

    effective_file_size = max(0, file_size - NO_TOUCH_ZONE)
    max_possible_samples = max(1, (effective_file_size - SAMPLE_LENGTH) // (SAMPLE_LENGTH // 2))
    num_samples = min(max_samples, max_possible_samples)
    
    samples = []
    for shard_id in range(num_samples):
        try:
            position = random.randint(NO_TOUCH_ZONE, max(NO_TOUCH_ZONE, file_size - SAMPLE_LENGTH))
            raw_sample = get_text_sample(file_path, position)
            if raw_sample is None:
                continue

            processed_sample = process_sample(raw_sample)
            if processed_sample is None or len(processed_sample) < SAMPLE_LENGTH / 2:
                logging.warning(f"Processed sample from {book_name} at position {position} is too short or failed.")
                continue  # Skip if the processed sample is too short
            
            samples.append((raw_sample, processed_sample))
        except Exception as e:
            logging.error(f"Error during sample extraction for {book_name} at position {position}: {e}")
            continue
    
    if samples:
        save_samples_to_csv(author, book_name, samples, sample_file)
        logging.info(f"Finished processing {file_path} with {len(samples)} samples.")
    else:
        logging.warning(f"No valid samples generated for book {book_name}.")
    
    return len(samples)

def parse_custom_id(custom_id):
    try:
        author_part, file_part = custom_id.split('-', 1)
        author_name = author_part.replace('_', '__')
        file_name = file_part.split('_', 1)[-1]  # Extract filename 
        return author_name, file_name
    except Exception as e:
        logging.error(f"Error parsing custom_id {custom_id}: {e}")
        return None, None



def load_eligible_books(jsonl_path):
    eligible_books = []  # A flat list to store all eligible books
    yes_count = 0  # Counter for entries with "YES" response

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    custom_id = entry['custom_id']
                    response_content = entry['response']['body']['choices'][0]['message']['content']
                    if response_content == "YES":
                        yes_count += 1  # Increment the counter
                        author_name, file_name = parse_custom_id(custom_id)
                        if author_name and file_name:
                            eligible_books.append((author_name, file_name))
                            logging.info(f"Eligible book: {author_name} -> {file_name}")
                        else:
                            logging.error(f"Failed to parse custom_id: {custom_id}")
                    else:
                        logging.info(f"Book with custom_id {custom_id} is marked as 'NO'. Skipping.")
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON line: {line}. Error: {e}")
                except KeyError as e:
                    logging.error(f"KeyError encountered in JSON entry {line}: {e}")
    except FileNotFoundError:
        logging.error(f"JSONL file not found: {jsonl_path}")
    except Exception as e:
        logging.error(f"Error loading eligible books from {jsonl_path}: {e}")
    
    return eligible_books, yes_count  # Return the count along with the eligible books





def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    sample_file = 'results.csv'

    logging.info("Loading eligible books...")
    eligible_books, yes_count = load_eligible_books(JSONL_PATH)

    if not eligible_books:
        logging.error("No eligible books found. Exiting.")
        return

    logging.info(f"Total eligible books detected: {len(eligible_books)}")
    logging.info(f"Total 'YES' responses in JSON: {yes_count}")

    # Pause and wait
    input("Press Enter to continue...")

    all_books = []

    try:
        for author_dir in os.listdir(BIG_TEXT_DIR):
            author_path = os.path.join(BIG_TEXT_DIR, author_dir)
            logging.info(f"Checking author directory: {author_dir}")
            if os.path.isdir(author_path):
                for book_file in os.listdir(author_path):
                    book_path = os.path.join(author_path, book_file)
                    if os.path.isfile(book_path) and os.path.getsize(book_path) >= MIN_BOOK_LENGTH + NO_TOUCH_ZONE:
                        # Check if is eligible by matching with the parsed JSONL file
                        if (author_dir, book_file) in eligible_books:
                            all_books.append((book_path, author_dir))
                            logging.info(f"Selected book: {book_file}")
                        else:
                            logging.info(f"Book {book_file} in author {author_dir} is not eligible.")
                    else:
                        logging.warning(f"File {book_file} in author {author_dir} is too short or not a valid file.")
            else:
                logging.warning(f"Author path {author_path} is not a directory.")
    except Exception as e:
        logging.error(f"Error collecting books from {BIG_TEXT_DIR}: {e}")

    if not all_books:
        logging.error("No books selected for processing. Exiting.")
        return

    logging.info(f"Total eligible books found: {len(all_books)}")

    random.shuffle(all_books)
    selected_books = all_books[:MAX_BOOKS]

    # args_list here
    args_list = []
    for book_path, author in selected_books:
        args_list.append((book_path, author, SAMPLES_PER_AUTHOR, sample_file))

    try:
        with open(sample_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['author', 'book', 'sample_id', 'raw_sample', 'processed_sample']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    except Exception as e:
        logging.error(f"Error initializing sample CSV file {sample_file}: {e}")
        return

    total_samples = 0
    for args in tqdm(args_list, total=len(args_list)):
        samples_processed = process_book(args)
        total_samples += samples_processed

    logging.info(f"Total samples processed: {total_samples}")

if __name__ == "__main__":
    main()