import pandas as pd
import numpy as np
import ast

def load_embeddings(embeddings_file, texts_file):

    embeddings_df = pd.read_csv(embeddings_file)
    texts_df = pd.read_csv(texts_file)
    combined_df = pd.merge(embeddings_df, texts_df, on='author', how='inner')
    embedding_columns = [col for col in combined_df.columns if col not in ['embedding_id', 'author', 'cleaned_text']]
    combined_df['embedding'] = combined_df[embedding_columns].values.tolist()
    return combined_df[['author', 'cleaned_text', 'embedding']]

def find_matching_text(original_text, adversarial_text, char_limit=500):
    original_substr = original_text[:char_limit]
    return original_substr in adversarial_text

def create_three_way_dataframe(normalized_adversarial_file, embeddings_file, texts_file, output_file):

    adversarial_df = pd.read_csv(normalized_adversarial_file)
    adversarial_df['generated_mimicry_embedding'] = adversarial_df['generated_mimicry_embedding'].apply(ast.literal_eval)
    original_data_df = load_embeddings(embeddings_file, texts_file)

    three_way_data = []

    for _, adv_row in adversarial_df.iterrows():
        author = adv_row['author']
        adversarial_text = adv_row['original_text']
        mimicry_embedding = adv_row['generated_mimicry_embedding']
        matching_original = original_data_df[original_data_df['author'] == author]
        
        match_found = False
        for _, orig_row in matching_original.iterrows():
            if find_matching_text(orig_row['cleaned_text'], adversarial_text):
                three_way_data.append({
                    'author': author,
                    'original_text': orig_row['cleaned_text'],
                    'original_embedding': orig_row['embedding'],
                    'adversarial_embedding': mimicry_embedding
                })
                match_found = True
                break
        
        if not match_found:
            print(f"Warning: No matching original text found for author {author}")
    three_way_df = pd.DataFrame(three_way_data)
    three_way_df.to_csv(output_file, index=False)
    print(f"Three-way dataframe saved to {output_file}")
    
    return three_way_df

normalized_adversarial_file = 'path/to/normalized_adversarial_embeddings.csv'
embeddings_file = 'path/to/original_embeddings.csv'
texts_file = 'path/to/original_texts.csv'
output_file = 'three_way_dataframe.csv'

three_way_df = create_three_way_dataframe(normalized_adversarial_file, embeddings_file, texts_file, output_file)

print(three_way_df.head())
unmatched_count = adversarial_df.shape[0] - three_way_df.shape[0]
print(f"Number of unmatched texts: {unmatched_count}")
