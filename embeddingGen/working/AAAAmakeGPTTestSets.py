import pandas as pd
import numpy as np
import ast
import os

def load_embeddings(file_path):

    df = pd.read_csv(file_path)
    for col in ['anchor_embedding', 'positive_embedding', 'negative_embedding']:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    return df

def create_adversarial_eval_set(normalized_adversarial_file, triplets_file, three_way_df_file, output_dir):
    adversarial_df = pd.read_csv(normalized_adversarial_file)
    adversarial_df['generated_mimicry_embedding'] = adversarial_df['generated_mimicry_embedding'].apply(ast.literal_eval)
    triplets_df = load_embeddings(triplets_file)

    three_way_df = pd.read_csv(three_way_df_file)
    three_way_df['original_embedding'] = three_way_df['original_embedding'].apply(ast.literal_eval)
    three_way_df['adversarial_embedding'] = three_way_df['adversarial_embedding'].apply(ast.literal_eval)
    new_eval_pairs = []

    for _, row in three_way_df.iterrows():
        original_embedding = row['original_embedding']
        adversarial_embedding = row['adversarial_embedding']
        matching_triplets = triplets_df[triplets_df['positive_embedding'].apply(
            lambda x: np.array_equal(x, original_embedding)
        )]

        for _, triplet in matching_triplets.iterrows():
            new_eval_pairs.append({
                'anchor_embedding': triplet['anchor_embedding'],
                'adversarial_embedding': adversarial_embedding
            })

    new_eval_df = pd.DataFrame(new_eval_pairs)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(normalized_adversarial_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_eval_set.csv")
    new_eval_df.to_csv(output_file, index=False)
    print(f"New evaluation set saved to: {output_file}")
    print(f"Total adversarial samples: {len(three_way_df)}")
    print(f"Total new evaluation pairs: {len(new_eval_df)}")

# Usage
normalized_adversarial_file = 'path/to/normalized_adversarial_embeddings.csv'
triplets_file = 'path/to/original_triplets.csv'
three_way_df_file = 'path/to/three_way_dataframe.csv'
output_dir = 'new_eval_sets'

create_adversarial_eval_set(normalized_adversarial_file, triplets_file, three_way_df_file, output_dir)
