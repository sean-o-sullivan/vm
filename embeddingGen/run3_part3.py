import os
import json
import random
import logging
from tqdm import tqdm


ROOT_DIRECTORY = '/home/aiadmin/Desktop/datasets/bigText'
WINDOW_SIZE = 900  
NO_TOUCH_ZONE = 5000  
BOOK_HARD_CAP = 3  


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SYSTEM_PROMPT = """Evaluate these texts for authorship verification studies. Exclude books with character dialogue. Accept only non-fiction, essays, and academic works with a consistent authorial voice. Output YES for suitable texts (where the author's unique style is prominent) or NO for unsuitable texts. Do not provide explanations."""

def get_text_sample(file_path, window_size, no_touch_zone):
    file_size = os.path.getsize(file_path)
    
    if file_size <= no_touch_zone + window_size:
        logging.warning(f"File {file_path} is too small for the specified no touch zone and window size. Skipping.")
        return None
    
    try:
        max_start_pos = file_size - (no_touch_zone + window_size)
        start_pos = no_touch_zone + random.randint(0, max_start_pos)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            file.seek(start_pos)
            return file.read(window_size)
    except (OSError, IOError) as e:
        logging.error(f"Failed to read {file_path}: {e}")
        return None

def create_batch_request(custom_id, content):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            "max_tokens": 5  
        }
    }

def process_corpus(root_dir, output_file):
    logging.info(f"Traversing root directory: {root_dir}")
    authors = [author for author in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, author))]
    logging.info(f"Found {len(authors)} authors.")

    total_books_processed = 0
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for author in tqdm(authors, desc="Processing authors"):
            author_dir = os.path.join(root_dir, author)
            books = [book for book in os.listdir(author_dir) if book.endswith('.txt')]
            logging.info(f"Processing {len(books)} books for author: {author}")
            
            for book in books:
                if total_books_processed >= BOOK_HARD_CAP:
                    logging.info(f"Reached hard cap of {BOOK_HARD_CAP} books. Stopping processing.")
                    return total_books_processed

                book_path = os.path.join(author_dir, book)
                sample = get_text_sample(book_path, WINDOW_SIZE, NO_TOUCH_ZONE)
                
                if sample:
                    custom_id = f"{author}-{book}"
                    batch_request = create_batch_request(custom_id, sample)
                    json.dump(batch_request, jsonl_file)
                    jsonl_file.write('\n')
                    total_books_processed += 1

    logging.info(f"Batch dataset created: {output_file}")
    logging.info(f"Total books processed: {total_books_processed}")
    return total_books_processed



if __name__ == "__main__":
    output_file = 'batch_dataset_classification.jsonl'
    logging.info("Starting the corpus processing for GPT-4 batch requests...")
    books_processed = process_corpus(ROOT_DIRECTORY, output_file)
    logging.info(f"Corpus processing completed. Processed {books_processed} books.")
