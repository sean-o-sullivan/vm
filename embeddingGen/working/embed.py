import csv
import random
import pandas as pd
import logging
from tqdm import tqdm
from embedding2 import generateEmbedding
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def process_csv(input_file, output_file_70, output_file_30):
#     # Read the CSV file and group texts by author
#     author_texts = {}
#     with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             author = row['author']
#             if author not in author_texts:
#                 author_texts[author] = []
#             author_texts[author].append(row)

#     # Get the list of authors and shuffle it
#     authors = list(author_texts.keys())
#     random.shuffle(authors)

#     # Calculate the split point
#     split_point = int(len(authors) * 0.7)

#     # Split authors into two groups
#     authors_70 = authors[:split_point]
#     authors_30 = authors[split_point:]

#     # Write to output files
#     write_output(output_file_70, author_texts, authors_70)
#     write_output(output_file_30, author_texts, authors_30)
    

def write_output(output_file, author_texts, authors):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(next(iter(author_texts.values()))[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for author in authors:
            for row in author_texts[author]:
                writer.writerow(row)

def process_entry(row, expected_keys, key_counts):
    author = row['author']
    # book_name = row.get('book', '')
    sample_id = row.get('sample_id', '')
    processed_sample = row['cleaned_text']
    
    processed_sample = processed_sample.replace("#/#\\#|||#/#\\#|||#/#\\#", "")
    
    print(f"Processing sample_id: {sample_id}")
    print(f"Processed sample (first 100 chars): {processed_sample[:100]}")
    try:
        embedding = generateEmbedding(processed_sample)
        current_keys = set(embedding.keys())
        new_keys = current_keys - expected_keys
        missing_keys = expected_keys - current_keys

        if new_keys:
            print(f"New keys found for sample_id {sample_id}: {new_keys}")
        if missing_keys:
            print(f"Missing keys for sample_id {sample_id}: {missing_keys}")

        for key in current_keys:
            key_counts[key] += 1
            new_row = {
            'author': author,
            # 'book': book_name,
            'sample_id': sample_id
        }
        new_row.update(embedding)
        return pd.Series(new_row)
    except Exception as e:
        logging.error(f"Error processing sample_id {sample_id}: {str(e)}")
        return None


def generate_embeddings(input_file, output_file):
    df = pd.read_csv(input_file)
    
    if df.empty:
        logging.error(f"No entries found in {input_file}. Skipping.")
        return
    
    print(f"CSV Headers: {df.columns.tolist()}")
    print(f"Total entries: {len(df)}")
    
    sample_text = """Gordon Edgley's sudden death came as a shock to everyone, not least himself. One moment he was in his study, seven words into the twenty-fifth sentence of the final chapter of his new book, And the Darkness Rained upon Them, and the next he was dead. A tragic loss, his mind echoed numbly as he slipped away.
The funeral was attended by family and acquaintances but not many friends. Gordon hadn't been a well-liked figure in the publishing world, for although the books he wrote—tales of horror and magic and wonder—regularly reared their heads in the bestseller lists, he had the disquieting habit of insulting people without realizing it, then laughing at their shock. It was at Gordon's funeral, however, that Stephanie Edgley first caught sight of the gentleman in the tan overcoat.
He was standing under the shade of a large tree, away from the crowd, the coat buttoned up all the way despite the warmth of the afternoon. A scarf was wrapped around the lower half of his face, and even from her position on the far side of the grave, Stephanie could make out the wild and frizzy hair that escaped from the wide-brimmed hat he wore low over his gigantic sunglasses. She watched him, intrigued by his appearance. And then, like he knew he was being observed, he turned and walked back through the rows of headstones and disappeared from sight.
After the service, Stephanie and her parents traveled back to her dead uncle's house, over a humpbacked bridge and along a narrow road that carved its way through thick woodland. The gates were heavy and grand and stood open, welcoming them into the estate. The grounds were vast, and the old house itself was ridiculously big.... There was an extra door in the living room, a door disguised as a bookcase, and when she was younger Stephanie liked to think that no one else knew about this door, not even Gordon himself. It was a secret passageway, like in the stories she'd read, and she'd make up adventures about haunted houses and smuggled treasures. This secret passageway would always be her escape route, and the imaginary villains in these adventures would be dumbfounded by her sudden and mysterious disappearance. But now this door, this secret passageway, stood open, and there was a steady stream of people through it, and she was saddened that this little piece of magic had been taken from her.
Tea was served and drinks were poured and little sandwiches were passed around on silver trays, and Stephanie watched the mourners casually appraise their surroundings. The major topic of hushed conversation was the will. Gordon wasn't a man who doted, or even demonstrated any great affection, so no one could predict who would inherit his substantial fortune. Stephanie could see the greed seep into the watery eyes of her father's other brother, a horrible little man called Fergus, as he nodded sadly and spoke somberly and pocketed the silverware when he thought no one was looking.
Fergus's wife was a thoroughly dislikable, sharp-featured woman named Beryl. She drifted through the crowd, deep in unconvincing grief, prying for gossip and digging for scandal. Her daughters did their best to ignore Stephanie. Carol and Crystal were twins, fifteen years old and as sour and vindictive as their parents. Whereas Stephanie was dark-haired, tall, slim, and strong, they were bottle blond, stumpy, and dressed in clothes that made them bulge in all the wrong places. Apart from their brown eyes, no one would have guessed that the twins were related to her. She liked that. It was the only thing about them she liked. She left them to their petty glares and snide whispers, and went for a walk.... The corridors of her uncle's house were long and lined with paintings. The floor beneath her feet was wooden, polished to a gleam, and the house smelled of age. Not musty, exactly, but . . . experienced. These walls and these floors had seen a lot in their time, and Stephanie was nothing but a faint whisper to them. Here one instant, gone the next."""
    sample_embedding = generateEmbedding(sample_text)
    expected_keys = set(sample_embedding.keys())
    print(f"Expected embedding keys: {expected_keys}")
    print(f"Expected embedding dimension: {len(expected_keys)}")
    key_counts = defaultdict(int)
    
    processed_rows = []
    counter = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing entries"):
        processed_row = process_entry(row, expected_keys, key_counts)
        if processed_row is not None:
            processed_rows.append(processed_row)
            counter += 1
        if counter % 20 == 0:
            temp_df = pd.DataFrame(processed_rows)
            temp_df.to_csv(output_file, index=False, mode='a', header=not pd.io.common.file_exists(output_file))
            processed_rows = []  
    if processed_rows:
        temp_df = pd.DataFrame(processed_rows)
        temp_df.to_csv(output_file, index=False, mode='a', header=not pd.io.common.file_exists(output_file))
    
    logging.info(f"Processing completed. Embeddings saved to {output_file}")



def main():
    # Process AGG.csv
   # process_csv('AGG.csv', 'AGG_70.csv', 'AGG_30.csv')
    generate_embeddings('AGG_70.csv', 'AGG_70_embeddings.csv')
    generate_embeddings('AGG_30.csv', 'AGG_30_embeddings.csv'
    process_csv('ABB.csv', 'ABB_70.csv', 'ABB_30.csv')
    generate_embeddings('ABB_70.csv', 'ABB_70_embeddings.csv')
    generate_embeddings('ABB_30.csv', 'ABB_30_embeddings.csv')

    print("Processing complete. All output files have been created.")

if __name__ == "__main__":
    main()
