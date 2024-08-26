import os
import csv
import stanza
import numpy as np
from tqdm import tqdm
from embedding2 import *


WINDOW_SIZE = 20  
ROOT_DIRECTORY = '/home/aiadmin/Desktop/datasets/bigText'
OUTPUT_CSV = 'corpus_embeddings.csv'


stanza.download('en')  
nlp = stanza.Pipeline('en', processors='tokenize')


def get_embedding(text):
    
    return your_embedding_module.get_embedding(text)


def process_book(file_path, author):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sentences]
    chunks = [' '.join(sentences[i:i+WINDOW_SIZE]) for i in range(0, len(sentences), WINDOW_SIZE)]
    
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append([author, os.path.basename(file_path)] + embedding)
    
    return embeddings


def process_corpus(root_dir, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Author', 'Book'] + [f'Feature_{i}' for i in range(len(get_embedding('Test')))])

        for author in tqdm(os.listdir(root_dir), desc="Processing authors"):
            author_dir = os.path.join(root_dir, author)
            if os.path.isdir(author_dir):
                for book in os.listdir(author_dir):
                    book_path = os.path.join(author_dir, book)
                    if os.path.isfile(book_path) and book.endswith('.txt'):
                        embeddings = process_book(book_path, author)
                        writer.writerows(embeddings)


if __name__ == "__main__":
    process_corpus(ROOT_DIRECTORY, OUTPUT_CSV)

