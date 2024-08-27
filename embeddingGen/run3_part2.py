import os
import csv
import random
import stanza
import numpy as np
import pandas as pd
from embedding2 import *
import re
import logging
from multiprocessing import Pool
from tqdm import tqdm

#a lot of capitals here, I know
ROOT_DIRECTORY = '/home/aiadmin/Desktop/datasets/bigText'
QUOTATION_THRESHOLD = 0.001008
BRACKETS_THRESHOLD = 0.002291
SAMPLES_PER_AUTHOR = 100
SAMPLE_LENGTH = 20
SAMPLE_BUFFER = 1.15


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


logging.info("Initializing Stanza pipeline...")
nlp = stanza.Pipeline('en', processors='tokenize')
logging.info("Stanza pipeline initialized.")


corpus_stats = pd.read_csv('corpus_statistics.csv')

def get_embedding(text):
    logging.debug(f"Generating embedding for text chunk: {text[:30]}...")
    embedding = generateEmbedding(text)
    logging.debug(f"Embedding generated with {len(embedding)} keys.")
    return embedding

def initialize_csv(output_file, sample_embedding):
    headers = ['Author', 'Book'] + list(sample_embedding.keys())
    logging.info(f"Initializing CSV file with headers: {headers}")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

def should_process_book(book_name, author):
    book_stats = corpus_stats[(corpus_stats['Book'] == book_name) & (corpus_stats['Author'] == author)]
    if book_stats.empty:
        logging.warning(f"No statistics found for {book_name} by {author}")
        return False
    
    quotation_density = book_stats['Quotations Rate per Character'].values[0]
    brackets_density = book_stats['Parentheticals Rate per Character'].values[0]
    
    return quotation_density < QUOTATION_THRESHOLD and brackets_density < BRACKETS_THRESHOLD

def get_sample_positions(file_size, num_samples):
    return [random.randint(0, file_size) for _ in range(num_samples)]

def estimate_sample_size(text):
    sentence_ends = len(re.findall(r'[.!?]+', text))
    return sentence_ends >= int(SAMPLE_LENGTH * SAMPLE_BUFFER)

def get_text_sample(file_path, start_pos):
    with open(file_path, 'r', encoding='utf-8') as file:
        file.seek(start_pos)
        sample = ""
        while not estimate_sample_size(sample):
            chunk = file.read(1000)
            if not chunk:
                break
            sample += chunk
    return sample

def process_sample(sample_text):
    doc = nlp(sample_text)
    sentences = [sentence.text for sentence in doc.sentences]
    return ' '.join(sentences[:SAMPLE_LENGTH])

def process_book(args):
    file_path, author, num_samples, output_file = args
    book_name = os.path.basename(file_path)
    logging.info(f"Processing book: {book_name} by {author}")
    
    if not should_process_book(book_name, author):
        return 0

    file_size = os.path.getsize(file_path)
    sample_positions = get_sample_positions(file_size, num_samples)
    
    embeddings = []
    for pos in sample_positions:
        raw_sample = get_text_sample(file_path, pos)
        processed_sample = process_sample(raw_sample)
        embedding = get_embedding(processed_sample)
        row = [author, book_name] + [embedding[key] for key in embedding]
        embeddings.append(row)
    
    logging.info(f"Saving {len(embeddings)} embeddings to {output_file}...")
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(embeddings)
    logging.info(f"Finished processing {file_path}.")
    return len(embeddings)

def process_author(author_dir, author, output_file):
    books = [book for book in os.listdir(author_dir) if book.endswith('.txt')]
    if not books:
        logging.warning(f"No books found for author: {author}")
        return

    samples_per_book = SAMPLES_PER_AUTHOR // len(books)
    extra_samples = SAMPLES_PER_AUTHOR % len(books)

    args_list = []
    for book in books:
        book_path = os.path.join(author_dir, book)
        samples_for_this_book = samples_per_book + (1 if extra_samples > 0 else 0)
        extra_samples -= 1
        args_list.append((book_path, author, samples_for_this_book, output_file))

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_book, args_list), total=len(args_list)))
    
    total_samples = sum(results)
    logging.info(f"Processed {total_samples} samples for {author}")

def process_corpus(root_dir):
    logging.info(f"Traversing root directory: {root_dir}")
    authors = [author for author in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, author))]
    logging.info(f"Found {len(authors)} authors.")

    for i, author in enumerate(authors):
        author_dir = os.path.join(root_dir, author)
        output_file = f'corpus_embeddings_core_{i+1}.csv'

        sample_text = "This is a sample text to initialize the CSV headers."
        sample_embedding = get_embedding(sample_text)
        initialize_csv(output_file, sample_embedding)

        process_author(author_dir, author, output_file)

    logging.info("Finished processing all authors.")

if __name__ == "__main__":
    logging.info("Starting the corpus processing...")
    process_corpus(ROOT_DIRECTORY)
    logging.info("Corpus processing completed.")
