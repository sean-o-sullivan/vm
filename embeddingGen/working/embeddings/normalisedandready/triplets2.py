import pandas as pd
from tqdm.auto import tqdm
import itertools
import numpy as np
from random import choice
import os
import ast

combination_value = 3 #virtual context size
virtual_text_limit = 5

def generate_all_combinations(df):
    all_combinations = []
    all_context_embeddings = []

    for author_id, group in tqdm(df.groupby('author'), desc='Processing authors', dynamic_ncols=True):
        print(f"Processing author ID: {author_id}, Number of texts: {len(group)}")

        author_texts = group.drop(columns=['embedding_id', 'author'])

        if len(author_texts) > virtual_text_limit:
            author_texts = author_texts.sample(n=virtual_text_limit)

        combinations, context_embeddings = generate_combinations_for_author(author_texts, author_id)

        all_combinations.extend(combinations)
        all_context_embeddings.extend(context_embeddings)

        print(f"Total combinations for this author: {len(combinations)}")

    return all_combinations, all_context_embeddings

def generate_combinations_for_author(author_texts, author_id):
    combinations_list = []
    context_embeddings_list = []
    for context_combination in itertools.combinations(author_texts.index, combination_value):
        context_embedding = author_texts.loc[list(context_combination)].mean().values.tolist()
        
        for positive_index in author_texts.index:
            if positive_index not in context_combination:
                positive_embedding = author_texts.loc[positive_index].values.tolist()
                combinations_list.append({
                    'anchor_embedding': context_embedding,
                    'positive_embedding': positive_embedding,
                    'author': author_id
                })
                context_embeddings_list.append(context_embedding)

    return combinations_list, context_embeddings_list

datasetpath = "GG_70.csv"
df = pd.read_csv(datasetpath)
of = os.path.splitext(datasetpath)[0]
suffix = of[-4:] if len(of) >= 4 else of

print("Author column type:", df['author'].dtype)

all_combinations, all_context_embeddings = generate_all_combinations(df)

combined_df = pd.DataFrame(all_combinations)

print(f"Total number of combinations: {len(combined_df)}")

authors = df['author'].unique()
false_pairs = []

for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc='Generating false pairs'):
    anchor_author = row['author']
    other_authors = [author for author in authors if author != anchor_author]
    
    if not other_authors:
        print(f"Warning: No other authors found for author {anchor_author}")
        continue
    
    negative_author = choice(other_authors)
    negative_texts = df[df['author'] == negative_author].drop(columns=['embedding_id', 'author'])
    negative_embed = negative_texts.sample(1).values.flatten().tolist()
    false_pairs.append(negative_embed)

combined_df['negative_embedding'] = false_pairs

def format_embedding(embedding):
    return ','.join(map(str, embedding))

combined_df['anchor_embedding'] = combined_df['anchor_embedding'].apply(format_embedding)
combined_df['positive_embedding'] = combined_df['positive_embedding'].apply(format_embedding)
combined_df['negative_embedding'] = combined_df['negative_embedding'].apply(format_embedding)

combined_df = combined_df.drop(columns=['author'])

print("\nFinal triplet structure:")
print(combined_df.head())
print(f"\nTotal triplets: {len(combined_df)}")
triplets_shuffled = combined_df.sample(frac=1).reset_index(drop=True)
output_file = f"Final-Triplets_{suffix}_|_VTL{virtual_text_limit}_C{combination_value}.csv"
triplets_shuffled.to_csv(output_file, index=False)

print(f"Triplet dataset is saved to {output_file}")
print("\nVerifying output...")
df_verify = pd.read_csv(output_file)
print(f"Output file shape: {df_verify.shape}")
print("First few rows of the output file:")
print(df_verify.head())

if df_verify.isnull().values.any():
    print("Warning: Output contains null values")
else:
    print("No null values found in the output")

if len(df_verify.columns) != 3:
    print(f"Warning: Expected 3 columns, but found {len(df_verify.columns)}")
else:
    print("Correct number of columns (3) in the output")
