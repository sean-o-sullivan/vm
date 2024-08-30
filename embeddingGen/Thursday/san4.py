import os
import csv
import random
import stanza
import logging
import json
from tqdm import tqdm
from collections import defaultdict
import re

SAMPLES_PER_AUTHOR = 100
SAMPLE_LENGTH = 20000  # Number of characters per sample
MIN_BOOK_LENGTH = SAMPLE_LENGTH * 2
NO_TOUCH_ZONE = 1000  # First 1000 characters will be skipped
MAX_BOOKS = 5000  # Maximum number of books to process in total

BIG_TEXT_DIR = '/home/aiadmin/Desktop/datasets/bigText'
JSONL_PATH = '/home/aiadmin/Desktop/code/vm/embeddingGen/batch_dataset_classification_output.jsonl'

try:
    nlp = stanza.Pipeline('en', processors='tokenize')
except Exception as e:
    logging.error(f"Failed to initialize Stanza pipeline: {e}")
    raise

def get_text_sample(file_path, position):
    encodings = ['utf-8', 'ISO-8859-1', 'windows-1252', 'ascii']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                file.seek(position)
                return file.read(SAMPLE_LENGTH)
        except UnicodeDecodeError:
            logging.warning(f"Failed to decode {file_path} with {encoding}. Trying next encoding.")
        except Exception as e:
            logging.error(f"Error reading text sample from {file_path} at position {position}: {e}")
            return None

    logging.error(f"Failed to read {file_path} with all attempted encodings.")
    return None

def process_sample(raw_sample):
    try:
        doc = nlp(raw_sample)
        sentences = [sent.text for sent in doc.sentences]

        # Drop the first and last sentences
        if len(sentences) > 2:
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
        logging.error(f"Failed to get size for file {file_path}: {e}")
        return 0

    if file_size < MIN_BOOK_LENGTH + NO_TOUCH_ZONE:
        logging.warning(f"File {book_name} is too short to process.")
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
        logging.warning(f"No valid samples were generated for book {book_name}.")

    return len(samples)

def parse_custom_id(custom_id):
    try:
        # Extract the filename part
        file_name = custom_id.split('-', 1)[-1]
        # Extract the unique ID and underscore from the start of the filename
        match = re.match(r'^(\d+_)', file_name)
        if match:
            return match.group(1)
        else:
            logging.error(f"Failed to extract unique ID from filename: {file_name}")
            return None
    except Exception as e:
        logging.error(f"Error parsing custom_id {custom_id}: {e}")
        return None

def load_eligible_books(jsonl_path):
    eligible_books = []
    yes_count = 0

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Processing JSONL entries"):
                try:
                    entry = json.loads(line)
                    custom_id = entry['custom_id']
                    response_content = entry['response']['body']['choices'][0]['message']['content']
                    if response_content == "YES":
                        yes_count += 1
                        book_id = parse_custom_id(custom_id)
                        if book_id:
                            eligible_books.append(book_id)
                            logging.info(f"Eligible book ID: {book_id}")
                            
                            # New: Stop after finding the corresponding entry
                            print(f"\nFound corresponding entry in JSONL file.")
                            print(f"Book ID to search for: {book_id}")
                            print(f"Search method: Will look for filenames starting with '{book_id}' (case-insensitive)")
                            input("Press Enter to continue searching for this file...")
                            
                            # Search for the file
                            found = False
                            for root, dirs, files in os.walk(BIG_TEXT_DIR):
                                for file in files:
                                    if file.lower().startswith(book_id.lower()):
                                        print(f"Found matching file: {os.path.join(root, file)}")
                                        found = True
                                        break
                                if found:
                                    break
                            
                            if not found:
                                print(f"No matching file found for {book_id}")
                            
                            input("Press Enter to continue processing the next entry...")
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

    return eligible_books, yes_count

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    sample_file = 'results.csv'

    logging.info("Loading eligible books...")
    eligible_books, yes_count = load_eligible_books(JSONL_PATH)

    if not eligible_books:
        logging.error("No eligible books found. Exiting.")
        return

    logging.info(f"Total eligible books detected: {len(eligible_books)}")
    for i in range(0,5):
        print(f"{eligible_books[i]}\n")

    logging.info(f"Total 'YES' responses in JSON: {yes_count}")

    # First halt
    input("Press Enter to begin matching files...")

    all_books = []
    found_books = set()

    try:
        for author_dir in os.listdir(BIG_TEXT_DIR):
            author_path = os.path.join(BIG_TEXT_DIR, author_dir)
            logging.info(f"Checking author directory: {author_dir}")
            if os.path.isdir(author_path):
                for book_file in os.listdir(author_path):
                    book_path = os.path.join(author_path, book_file)
                    logging.info(f"Attempting to match: Book = {book_file}")

                    # Check if the book_file starts with any eligible book ID and contains an underscore
                    matching_books = [book_id for book_id in eligible_books if book_file.lower().startswith(book_id.lower())]

                    if matching_books:
                        if os.path.isfile(book_path) and os.path.getsize(book_path) >= MIN_BOOK_LENGTH + NO_TOUCH_ZONE:
                            all_books.append((book_path, author_dir))
                            found_books.add(book_file.lower())
                            logging.info(f"Matched and selected book: {book_file} under author {author_dir}")
                        else:
                            logging.warning(f"File {book_file} is too short or not a valid file.")
                    else:
                        logging.info(f"No match found for book {book_file}.")
            else:
                logging.warning(f"Author path {author_path} is not a directory.")
    except Exception as e:
        logging.error(f"Error collecting books from {BIG_TEXT_DIR}: {e}")

    if not all_books:
        logging.error("No books selected for processing. Exiting.")
        return

    logging.info(f"Total eligible books found: {len(all_books)}")

    # Second halt
    input("Press Enter to start processing the selected books...")

    # Find books that were not found in the corpus
    not_found_books = set(book.lower() for book in eligible_books) - found_books

    # Third halt and print not found books
    print(f"\nNumber of books not found in the corpus: {len(not_found_books)}")
    input("Press Enter to see the list of books not found in the corpus...")

    print("\nBooks not found in the corpus:")
    for book in sorted(not_found_books):
        print(book)

    random.shuffle(all_books)
    selected_books = all_books[:MAX_BOOKS]

    # Define args_list here
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
    for args in tqdm(args_list, total=len(args_list), desc="Processing books"):
        samples_processed = process_book(args)
        total_samples += samples_processed

    logging.info(f"Total samples processed: {total_samples}")

if __name__ == "__main__":
    main()
