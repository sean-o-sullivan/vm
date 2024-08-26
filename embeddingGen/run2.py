import os
import csv
import stanza
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from embedding2 import *


WINDOW_SIZE = 20  
ROOT_DIRECTORY = '/home/aiadmin/Desktop/datasets/bigText'
NUM_PROCESSES = 8  


stanza.download('en')  
nlp = stanza.Pipeline('en', processors='tokenize')


def get_embedding(text):
    
    print(text,"recieved by the get_embedding function")
    return generateEmbedding(text)


def process_book(file_path_author):
    file_path, author, output_file = file_path_author
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sentences]
    chunks = [' '.join(sentences[i:i+WINDOW_SIZE]) for i in range(0, len(sentences), WINDOW_SIZE)]
    
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append([author, os.path.basename(file_path)] + embedding)
    
    
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(embeddings)


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def process_corpus(root_dir, num_processes):
    authors = [author for author in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, author))]

    
    author_chunks = chunkify(authors, num_processes)

    
    tasks = []
    for i, author_chunk in enumerate(author_chunks):
        output_file = f'corpus_embeddings_core_{i+1}.csv'
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Author', 'Book'] + [f'Feature_{j}' for j in range(len(get_embedding('Test')))])

        for author in author_chunk:
            author_dir = os.path.join(root_dir, author)
            for book in os.listdir(author_dir):
                book_path = os.path.join(author_dir, book)
                if os.path.isfile(book_path) and book.endswith('.txt'):
                    tasks.append((book_path, author, output_file))

    
    with Pool(processes=num_processes) as pool:
        
        list(tqdm(pool.imap(process_book, tasks), total=len(tasks), desc="Processing books"))


if __name__ == "__main__":
    process_corpus(ROOT_DIRECTORY, NUM_PROCESSES)


