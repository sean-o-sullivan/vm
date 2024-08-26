import os
import csv
import stanza
import numpy as np
from embedding2 import *


WINDOW_SIZE = 20
ROOT_DIRECTORY = '/home/aiadmin/Desktop/datasets/bigText'


print("Downloading and initializing ze ze Stanza pipeline...")
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize')
print("Stanza pipeline initialized.")

def get_embedding(text):
    print(f"Generating embedding for text chunk: {text[:30]}...")
    embedding = generateEmbedding(text)
    print(f"Embedding generated for text chunk of length {len(text)}.")
    return embedding

def process_book(file_path, author, output_file):
    print(f"Processing book: {file_path} by {author}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        print(f"Text of length {len(text)} read from {file_path}")

    print("Passing text to Stanza pipeline for tokenization...")
    doc = nlp(text)  
    print(f"Tokenization completed for {file_path}")

    sentences = [sentence.text for sentence in doc.sentences]
    chunks = [' '.join(sentences[i:i+WINDOW_SIZE]) for i in range(0, len(sentences), WINDOW_SIZE)]
    print(f"Book split into {len(chunks)} chunks.")

    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        embedding = get_embedding(chunk)
        embeddings.append([author, os.path.basename(file_path)] + embedding)
    
    print(f"Saving embeddings to {output_file}...")
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(embeddings)
    print(f"Finished processing {file_path}.")

def process_corpus(root_dir):
    print(f"Traversing root directory: {root_dir}")
    authors = [author for author in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, author))]
    print(f"Found {len(authors)} authors.")

    for i, author in enumerate(authors):
        output_file = f'corpus_embeddings_core_{i+1}.csv'
        print(f"Initializing CSV file: {output_file}")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Author', 'Book'] + [f'Feature_{j}' for j in range(111)])

        author_dir = os.path.join(root_dir, author)
        for book in os.listdir(author_dir):
            book_path = os.path.join(author_dir, book)
            if os.path.isfile(book_path) and book.endswith('.txt'):
                process_book(book_path, author, output_file)

    print("Finished processing all books. :)")

if __name__ == "__main__":
    print("Starting the corpus processing...")
    process_corpus(ROOT_DIRECTORY)
    print("Corpus processing completed.")
