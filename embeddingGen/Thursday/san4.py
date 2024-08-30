import os
import csv
import random
import stanza
import logging
import json
from tqdm import tqdm
import re

SAMPLES_PER_AUTHOR = 10
SAMPLE_LENGTH = 10000  # Number of characters per sample
MIN_BOOK_LENGTH = SAMPLE_LENGTH * 2
NO_TOUCH_ZONE = 1000  # First 1000 characters will be skipped
MAX_BOOKS = 5000  # Maximum number of books to process in total

BIG_TEXT_DIR = '/home/aiadmin/Desktop/datasets/bigText'
JSONL_PATH = '/home/aiadmin/Desktop/code/vm/embeddingGen/batch_dataset_classification_output.jsonl'
CSV_PATH = 'file_author_map.csv'  # CSV file to store file paths and authors

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
    delimiter = "#/#\#|||#/#\#|||#/#\#"
    
    try:
        with open(sample_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['author', 'book', 'sample_id', 'raw_sample', 'processed_sample']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            for i, (raw_sample, processed_sample) in enumerate(samples):
                # Insert the custom delimiter between raw and processed samples
                processed_sample_with_delimiter = f"{raw_sample}{delimiter}{processed_sample}"
                
                writer.writerow({
                    'author': author,
                    'book': book_name,
                    'sample_id': i,
                    'processed_sample': processed_sample_with_delimiter
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

    if file_size < MIN_BOOK_LENGTH + 2 * NO_TOUCH_ZONE:  # Adjusted to account for the no touch zone at both ends
        logging.warning(f"File {book_name} is too short to process.")
        return 0

    # Effective size excluding the no-touch zones at the beginning and end
    effective_file_size = file_size - 2 * NO_TOUCH_ZONE
    max_possible_samples = max(1, (effective_file_size - SAMPLE_LENGTH) // (SAMPLE_LENGTH // 2))
    num_samples = min(max_samples, max_possible_samples)

    samples = []
    for shard_id in range(num_samples):
        try:
            position = random.randint(NO_TOUCH_ZONE, file_size - NO_TOUCH_ZONE - SAMPLE_LENGTH)
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
    file_author_map = []

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
                            
                            # Search for the file
                            found = False
                            for root, dirs, files in os.walk(BIG_TEXT_DIR):
                                for file in files:
                                    if file.lower().startswith(book_id.lower()):
                                        file_author_map.append((os.path.join(root, file), os.path.basename(root)))
                                        logging.info(f"Found matching file: {os.path.join(root, file)}")
                                        found = True
                                        break
                                if found:
                                    break
                            
                            if not found:
                                logging.warning(f"No matching file found for {book_id}")
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

    return file_author_map, yes_count

def save_file_author_map(file_author_map, csv_path):
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file_path', 'author']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for file_path, author in file_author_map:
                writer.writerow({'file_path': file_path, 'author': author})
    except Exception as e:
        logging.error(f"Error saving file-author map to CSV: {e}")

def load_file_author_map(csv_path):
    file_author_map = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                file_author_map.append((row['file_path'], row['author']))
    except Exception as e:
        logging.error(f"Error loading file-author map from CSV: {e}")
    return file_author_map

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    sample_file = 'results_10KSample.csv'

    if os.path.exists(CSV_PATH):
        logging.info(f"Loading file-author map from {CSV_PATH}")
        file_author_map = load_file_author_map(CSV_PATH)
    else:
        logging.info("Loading eligible books...")
        file_author_map, yes_count = load_eligible_books(JSONL_PATH)

        if not file_author_map:
            logging.error("No eligible books found. Exiting.")
            return

        logging.info(f"Total eligible books detected: {len(file_author_map)}")
        logging.info(f"Total 'YES' responses in JSON: {yes_count}")

        save_file_author_map(file_author_map, CSV_PATH)

    random.shuffle(file_author_map)
    selected_books = file_author_map[:MAX_BOOKS]

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
