import pandas as pd
from tqdm.auto import tqdm
import itertools
from math import comb
import numpy as np
from random import choice
import os

combination_value = 3
virtual_text_limit = 5

def generate_all_combinations(df):
    all_combinations = []
    all_context_embeddings = []
    context_index = 0

    for author_id, group in tqdm(df.groupby('author'), desc='Processing authors', dynamic_ncols=True):
        print(f"Processing author ID: {author_id}, Number of texts: {len(group)}")

        author_texts = group.drop(columns=['embedding_id', 'author'])

        if len(author_texts) > virtual_text_limit:
            author_texts = author_texts.sample(n=virtual_text_limit)

        combinations, context_embeddings = generate_combinations_for_author(author_texts, author_id, context_index)

        all_combinations.extend(combinations)
        all_context_embeddings.extend(context_embeddings)
        context_index += len(context_embeddings)

        print(f"Total combinations for this author: {len(combinations)}")

    return all_combinations, all_context_embeddings

def generate_combinations_for_author(author_texts, author_id, start_context_index):
    combinations_list = []
    context_embeddings_list = []
    context_index = start_context_index

    for context_combination in itertools.combinations(author_texts.index, combination_value):
        context_embedding = author_texts.loc[list(context_combination)].mean().values.tolist()
        context_embeddings_list.append(context_embedding + [author_id])

        for positive_index in author_texts.index:
            if positive_index not in context_combination:
                positive_embedding = author_texts.loc[positive_index].values.tolist() + [author_id]
                combinations_list.append([context_index, positive_embedding])

        context_index += 1

    return combinations_list, context_embeddings_list

datasetpath = "GG_30.csv"
df = pd.read_csv(datasetpath)

of = os.path.splitext(datasetpath)[0]
suffix = of[-4:] if len(of) >= 4 else of

if df['author'].dtype == 'object':
    print("Author column is of type string")
else:
    print("Author column is of type numeric")

all_combinations, all_context_embeddings = generate_all_combinations(df)

combinations_df = pd.DataFrame(all_combinations, columns=['context_index', 'positive_embedding'])
context_embeddings_df = pd.DataFrame(all_context_embeddings, 
                                     columns=['context_index'] + [str(i) for i in range(len(all_context_embeddings[0]) - 2)] + ['author'])

combinations_df['positive_embedding'] = combinations_df['positive_embedding'].apply(lambda x: str(x[:-1]))
combinations_df = combinations_df.drop('context_index', axis=1)

context_embeddings_df['anchor_embedding'] = context_embeddings_df.iloc[:, 1:-1].apply(lambda row: str(row.tolist()), axis=1)
context_embeddings_df = context_embeddings_df.drop(['context_index', 'author'], axis=1)

combined_df = pd.concat([context_embeddings_df['anchor_embedding'], combinations_df], axis=1)

false_pairs_list = []
authors = df['author'].unique()

for index, context_row in tqdm(context_embeddings_df.iterrows(), total=context_embeddings_df.shape[0], desc='Generating false pairs', dynamic_ncols=True):
    anchor_embedding = context_row['anchor_embedding']
    
    
    anchor_author = all_context_embeddings[index][-1]
    
    
    other_authors = [author for author in authors if author != anchor_author]
    if not other_authors:
        print(f"Warning: No other authors found for context embedding at index {index}")
        continue
    
    negative_author = choice(other_authors)
    
    
    negative_texts = df[df['author'] == negative_author].drop(columns=['embedding_id', 'author'])
    negative_embed = negative_texts.sample(1).values.flatten().tolist()
    
    false_pairs_list.append({
        'anchor_embedding': anchor_embedding,
        'negative_embedding': str(negative_embed)
    })

false_pairs_df = pd.DataFrame(false_pairs_list)

triplets = pd.concat([combined_df, false_pairs_df['negative_embedding']], axis=1)

print("\nThe triplet has the following columns:", triplets.columns)
print(triplets.head())

triplets_shuffled = triplets.sample(frac=1).reset_index(drop=True)

triplets_shuffled.to_csv(f"Final-Triplets_{suffix}_|2|_VTL{virtual_text_limit}_C{combination_value}.csv", index=False)

print("Triplet dataset is now saved!")
