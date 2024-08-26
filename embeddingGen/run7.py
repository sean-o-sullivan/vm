import os
import csv
from tqdm import tqdm

ROOT_DIRECTORY = '/home/aiadmin/Desktop/datasets/bigText'

def quotations_rate(text, delimiters={'"', "'", '“', '”', '‘', '’', '«', '»', '‹', '›'}):
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    
    delimiter_count = sum(1 for char in text if char in delimiters)
    return delimiter_count / total_chars

def parentheticals_and_brackets_rate(text, delimiters={'(', ')', '[', ']'}):
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    
    delimiter_count = sum(1 for char in text if char in delimiters)
    return delimiter_count / total_chars

def process_book(file_path, author, output_file):
   # print(f"Processing book: {file_path} by {author}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
    #        print(f"Text of length {len(text)} read from {file_path}")
    except Exception as e:
    #    print(f"Error reading {file_path}: {e}")
        return

    quotations_rate_value = quotations_rate(text)
    parentheticals_rate_value = parentheticals_and_brackets_rate(text)

    row = [author, os.path.basename(file_path), quotations_rate_value, parentheticals_rate_value]
    
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

def initialize_csv(output_file):
    headers = ['Author', 'Book', 'Quotations Rate per Character', 'Parentheticals Rate per Character']
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

def process_corpus(root_dir):
    print(f"Traversing root directory: {root_dir}")
    authors = [author for author in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, author))]
    print(f"Found {len(authors)} authors.")

    output_file = 'corpus_statistics.csv'
    initialize_csv(output_file)

    total_books = 0
    for author in authors:
        author_dir = os.path.join(root_dir, author)
        total_books += sum(1 for book in os.listdir(author_dir) if os.path.isfile(os.path.join(author_dir, book)) and book.endswith('.txt'))

    with tqdm(total=total_books, desc="Processing books", unit="book") as pbar:
        for author in authors:
            author_dir = os.path.join(root_dir, author)
            for book in os.listdir(author_dir):
                book_path = os.path.join(author_dir, book)
                if os.path.isfile(book_path) and book.endswith('.txt'):
                    process_book(book_path, author, output_file)
                    pbar.update(1)

    print("Finished processing all books.")

if __name__ == "__main__":
    print("Starting the corpus processing...")
    process_corpus(ROOT_DIRECTORY)
    print("Corpus processing completed.")
