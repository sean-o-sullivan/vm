import json
import csv
import logging
from tqdm import tqdm


def process_jsonl(input_file, output_file):
    suitable_books = []
    total_books = 0
    suitable_count = 0

    logging.info(f"Processing input file: {input_file}")
    
    with open(input_file, 'r') as jsonl_file:
        for line in tqdm(jsonl_file, desc="Processing books"):
            total_books += 1
            data = json.loads(line)
            
            custom_id = data['custom_id']
            response = data['response']['body']['choices'][0]['message']['content']
            
            # if the response is 'YES'
            if response.strip().upper() == 'YES':
                author, book = custom_id.rsplit('-', 1)  # Split from the right side
                suitable_books.append([author, book])
                suitable_count += 1

    logging.info(f"Total books processed: {total_books}")
    logging.info(f"Suitable books found: {suitable_count}")

    logging.info(f"Writing suitable books to: {output_file}")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Author', 'Book'])
        writer.writerows(suitable_books)

    logging.info("Processing completed.")

if __name__ == "__main__":
    input_file = 'batch_dataset_classification_output_5.jsonl'
    output_file = 'suitable_books_for_authorship_verification.csv'
    process_jsonl(input_file, output_file)
