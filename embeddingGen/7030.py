import csv
import random

def process_csv(input_file, output_file_70, output_file_30):

    author_texts = {}
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            author = row['author']
            if author not in author_texts:
                author_texts[author] = []
            author_texts[author].append(row)

    authors = list(author_texts.keys())
    random.shuffle(authors)
    split_point = int(len(authors) * 0.7)
    authors_70 = authors[:split_point]
    authors_30 = authors[split_point:]
    write_output(output_file_70, author_texts, authors_70)
    write_output(output_file_30, author_texts, authors_30)

def write_output(output_file, author_texts, authors):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(next(iter(author_texts.values()))[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for author in authors:
            for row in author_texts[author]:
                writer.writerow(row)

process_csv('AGG.csv', 'AGG_70.csv', 'AGG_30.csv')
process_csv('ABB.csv', 'ABB_70.csv', 'ABB_30.csv')

print("Processing complete. Output files have been created with no overlap.")
