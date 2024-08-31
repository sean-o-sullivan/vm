import csv
import logging
from tqdm import tqdm
from embedding2 import generateEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_entry(entry, fieldnames, output_file):
    author = entry['author']
    book_name = entry['book']
    sample_id = entry['sample_id']
    processed_sample = entry['processed_sample']
    print(f"This is the processed_sample: {processed_sample}")

    
    embedding = generateEmbedding(processed_sample)

    row = {
        'author': author,
        'book': book_name,
        'sample_id': sample_id
    }
    row.update(embedding)  

    
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

def main():
    input_csv = '/home/aiadmin/Desktop/code/vm/embeddingGen/Thursday/results_10KSample.csv'
    output_file = 'output_embeddings.csv'

    
    with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        entries = list(reader)
    
    if not entries:
        logging.error("No entries found in results.csv. Exiting.")
        return

    textn = """
Gordon Edgley's sudden death came as a shock to everyone—not least himself. One moment he was in his study, seven words into the twenty-fifth sentence of the final chapter of his new book, And the Darkness Rained upon Them, and the next he was dead. A tragic loss, his mind echoed numbly as he slipped away.

The funeral was attended by family and acquaintances but not many friends. Gordon hadn't been a well-liked figure in the publishing world, for although the books he wrote—tales of horror and magic and wonder—regularly reared their heads in the bestseller lists, he had the disquieting habit of insulting people without realizing it, then laughing at their shock. It was at Gordon's funeral, however, that Stephanie Edgley first caught sight of the gentleman in the tan overcoat.

He was standing under the shade of a large tree, away from the crowd, the coat buttoned up all the way despite the warmth of the afternoon. A scarf was wrapped around the lower half of his face, and even from her position on the far side of the grave, Stephanie could make out the wild and frizzy hair that escaped from the wide-brimmed hat he wore low over his gigantic sunglasses. She watched him, intrigued by his appearance. And then, like he knew he was being observed, he turned and walked back through the rows of headstones and disappeared from sight[1].

The excerpt continues to describe Stephanie and her parents visiting her uncle's house after the funeral, the secret bookcase door, and the interactions with other family members at the gathering. It provides vivid details about the setting and characters, introducing readers.
"""
    sample_embedding = generateEmbedding(textn)
    fieldnames = ['author', 'book', 'sample_id'] + list(sample_embedding.keys())

    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    
    for entry in tqdm(entries, desc="Processing entries"):
        process_entry(entry, fieldnames, output_file)
        input("press enter to continue")

    logging.info(f"Processing completed. Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()
