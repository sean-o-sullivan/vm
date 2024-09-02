import pandas as pd
import numpy as np

def find_embedding_column(df):
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    if not embedding_cols:
        raise ValueError(f"No embedding column found in the df. Columns: {df.columns}")
    if len(embedding_cols) > 1:
        print(f"Warning: Multiple embedding columns found: {embedding_cols}. Using the first one.")
    return embedding_cols[0]

def load_csv(file_path, first_file=False):
    
    print(f"Loading the file: {file_path}")
    df = pd.read_csv(file_path)
    embedding_col = find_embedding_column(df)
    
    if first_file:
        return df[['author', 'original_text', embedding_col]]
    else:
        return df[[embedding_col]]

def consolidate_embeddings(file_paths, output_file):

    consolidated_df = None
    
    for i, file_path in enumerate(file_paths):
        if i == 0:
            consolidated_df = load_csv(file_path, first_file=True)
            consolidated_df.rename(columns={find_embedding_column(consolidated_df): f'embedding_{i}'}, inplace=True)
        else:
            df = load_csv(file_path)
            embedding_col = find_embedding_column(df)
            consolidated_df[f'embedding_{i}'] = df[embedding_col]
    
    print(f"Saving consolidated data to {output_file}")
    print(f"Final shape: {consolidated_df.shape}")
    print(f"Columns: {consolidated_df.columns.tolist()}")
    for col in consolidated_df.columns:
        if 'embedding' in col.lower():
            consolidated_df[col] = consolidated_df[col].apply(lambda x: ','.join(map(str, eval(x))) if isinstance(x, str) else ','.join(map(str, x)))
    
    consolidated_df.to_csv(output_file, index=False)
    print("Consolidation complete.")

if __name__ == "__main__":
    file_paths = [
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/BB_30.csv',
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT3ABB_30_embeddings.csv',
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT3ABB_30_embeddings.csv',
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4TABB_30_embeddings.csv',
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4TABB_30_embeddings.csv',
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4oABB_30_embeddings.csv',
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4TABB_30_embeddings.csv'
    ]

    output_file = "consolidated_embeddings.csv"
    consolidate_embeddings(file_paths, output_file)
