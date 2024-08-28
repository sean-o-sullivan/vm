import os
import csv
import random
import stanza
import logging
from multiprocessing import Pool
from tqdm import tqdm
from embedding2 import * 
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
    
    # Drop the first and last sentences
    if len(sentences) > 2:
        processed_sample = ' '.join(sentences[1:-1])
    else:
        processed_sample = ''
    
    return processed_sample

def process_book(args):
    file_path, author, max_samples, output_file, fieldnames = args
    book_name = os.path.basename(file_path)
    logging.info(f"Processing book: {book_name} by {author}")

    file_size = os.path.getsize(file_path)
    effective_file_size = max(0, file_size - NO_TOUCH_ZONE)
    max_possible_samples = max(1, (effective_file_size - SAMPLE_LENGTH) // (SAMPLE_LENGTH // 2))
    num_samples = min(max_samples, max_possible_samples)
    
    embeddings = []
    for shard_id in range(num_samples):
        position = random.randint(NO_TOUCH_ZONE, max(NO_TOUCH_ZONE, file_size - SAMPLE_LENGTH))
        raw_sample = get_text_sample(file_path, position)
        processed_sample = process_sample(raw_sample)
        
        if len(processed_sample) < SAMPLE_LENGTH / 2:
            continue  
        
        embedding = generateEmbedding(processed_sample)
        row = {
            'author': author,
            'book': book_name,
            'shardID': shard_id
        }
        row.update(embedding)  # Add all key-value pairs from the embedding dictionary
        embeddings.append(row)
    
    logging.info(f"Saving {len(embeddings)} embeddings to {output_file}...")
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
                if os.path.isfile(book_path) and os.path.getsize(book_path) >= MIN_BOOK_LENGTH + NO_TOUCH_ZONE:
                    all_books.append((book_path, author_dir))

    # Shuffle and limit to MAX_BOOKS
    random.shuffle(all_books)
    selected_books = all_books[:MAX_BOOKS]

    sampleText = """
Chapter One of "Skulduggery Pleasant" by Derek Landy begins with the sudden death of Gordon Edgley, which comes as a shock to everyone, including himself[1]. The story then moves to Gordon's funeral, where his niece Stephanie Edgley first notices a mysterious gentleman in a tan overcoat[1].

The funeral is described as being attended by family and acquaintances, but not many friends, as Gordon wasn't well-liked in the publishing world despite his successful horror and magic books[1]. After the service, Stephanie and her parents travel to Gordon's house, which is described as ridiculously big with vast grounds[1].

At the house, Stephanie observes the mourners during the post-funeral gathering. She notes the greed in her uncle Fergus's eyes as he pockets silverware, and describes her aunt Beryl as a dislikable woman prying for gossip[1]. Stephanie's cousins, Carol and Crystal, are portrayed as sour and vindictive twins who ignore her[1].

The chapter also mentions a secret door in the living room disguised as a bookcase, which Stephanie used to imagine as part of her childhood adventures. However, during the gathering, this door stands open, disappointing Stephanie as it loses its magical quality[1].

The excerpt ends with Stephanie taking a walk through the corridors of her uncle's house, describing the long hallways lined with paintings, the polished wooden floors, and the overall sense of age and experience that permeates the house[1].
 """
    sample_embedding = generateEmbedding(sampleText)
    fieldnames = ['author', 'book', 'shardID'] + list(sample_embedding.keys())

    args_list = []
    for book_path, author in selected_books:
        args_list.append((book_path, author, SAMPLES_PER_AUTHOR, output_file, fieldnames))

    # Initialize the CSV file with headers
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_book, args_list), total=len(args_list)))

    total_samples = sum(results)
    logging.info(f"Total samples processed: {total_samples}")

if __name__ == "__main__":
    main()
