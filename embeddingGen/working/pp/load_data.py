import pandas as pd
import numpy as np
from utils import save_to_csv

def load_embeddings(embeddings_file, texts_file):

    print(f"Loading embeddings from {embeddings_file}")
    embeddings_df = pd.read_csv(embeddings_file)
    print(f"Embeddings shape: {embeddings_df.shape}")
    print(f"Loading texts from {texts_file}")
    texts_df = pd.read_csv(texts_file)
    print(f"Texts shape: {texts_df.shape}")
    merged_df = pd.merge(embeddings_df, texts_df, on='author')
    print(f"Merged dataframe shape: {merged_df.shape}")
    save_to_csv(merged_df, 'original_data.csv')
    return merged_df

def load_adversarial_embeddings(file_path, embedding_col):

    print(f"Loading adversarial embeddings from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataframe shape before processing: {df.shape}")
    df[embedding_col] = df[embedding_col].apply(eval).apply(np.array)
    print(f"Dataframe shape after processing: {df.shape}")
    return df

def load_all_data(original_embeddings_file, original_texts_file, adversarial_files):

    print("Loading all data")
    data = {
        'original': load_embeddings(original_embeddings_file, original_texts_file)
    }
    
    for adv_type, file_path in adversarial_files.items():
        print(f"Processing {adv_type} data")
        embedding_col = 'generated_mimicry_embedding' if 'mimic' in adv_type else 'generated_text_embedding'
        data[adv_type] = load_adversarial_embeddings(file_path, embedding_col)
        save_to_csv(data[adv_type], f'{adv_type}_data.csv')
    
    print("All data is loaded successfully")
    return data
