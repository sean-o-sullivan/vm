import os
import csv
import stanza


ROOT_DIRECTORY = '/home/aiadmin/Desktop/datasets/bigText'


print("Downloading and initializing Stanza pipeline...")
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize')
print("Stanza pipeline initialized.")

def count_sentences(text):

    doc = nlp(text)
    return len(doc.sentences)

def quotations_rate(text, delimiters={'"', "'", '“', '”', '‘', '’', '«', '»', '‹', '›'}):

    sentence_count = count_sentences(text)
    if sentence_count == 0:
        return 0.0
    
    delimiter_count = sum(1 for char in text if char in delimiters)
    return delimiter_count / sentence_count

def parentheticals_and_brackets_rate(text, delimiters={'(', ')', '[', ']'}):
 
    sentence_count = count_sentences(text)
    if sentence_count == 0:
        return 0.0
    
    delimiter_count = sum(1 for char in text if char in delimiters)
    return delimiter_count / sentence_count

def process_book(file_path, author, output_file):
 
    print(f"Processing book: {file_path} by {author}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        print(f"Text of length {len(text)} read from {file_path}")

    quotations_rate_value = quotations_rate(text)
    parentheticals_rate_value = parentheticals_and_brackets_rate(text)

    row = [author, os.path.basename(file_path), quotations_rate_value, parentheticals_rate_value]
    
    print(f"Saving results to {output_file}...")
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)
    print(f"Finished processing {file_path}.")

def initialize_csv(output_file):
  
    headers = ['Author', 'Book', 'Quotations Rate per Sentence', 'Parentheticals Rate per Sentence']
    print(f"Initializing CSV file with headers: {headers}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

def process_corpus(root_dir):
   
    print(f"Traversing root directory: {root_dir}")
    authors = [author for author in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, author))]
    print(f"Found {len(authors)} authors.")

    output_file = 'corpus_statistics.csv'
    initialize_csv(output_file)

    for author in authors:
        author_dir = os.path.join(root_dir, author)

        for book in os.listdir(author_dir):
            book_path = os.path.join(author_dir, book)
            if os.path.isfile(book_path) and book.endswith('.txt'):
                process_book(book_path, author, output_file)

    print("Finished processing all books.")

if __name__ == "__main__":
    print("Starting the corpus processing...")
    process_corpus(ROOT_DIRECTORY)
    print("Corpus processing completed.")
