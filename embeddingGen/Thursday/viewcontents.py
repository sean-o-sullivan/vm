import json
import os

INPUT_FILE = '/home/aiadmin/Desktop/code/vm/embeddingGen/batch_dataset_classification_output.jsonl'
BOOKS_DIR = '/home/aiadmin/Desktop/datasets/bigText'
PREVIEW_LENGTH = 5000

def get_book_preview(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read(PREVIEW_LENGTH)
    except FileNotFoundError:
        return f"File not found: {file_path}"

def main():
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            custom_id = data['custom_id']
            response = data['response']['body']['choices'][0]['message']['content']
            
            if response == "YES":
                author, book = custom_id.split('-', 1)
                book = book.split('_', 1)[1]  # Remove the number prefix
                file_path = os.path.join(BOOKS_DIR, author, custom_id)
                
                print(f"\nBook: {book}")
                print(f"Author: {author}")
                print(f"Preview:\n{get_book_preview(file_path)}")
                
                input("Press Enter to continue to the next book...")

if __name__ == "__main__":
    main()
