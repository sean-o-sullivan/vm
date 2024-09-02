import pandas as pd
import numpy as np
import ast

def find_embedding_column(df):
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    if not embedding_cols:
        raise ValueError(f"No embedding column found in the df. Columns: {df.columns}")
    if len(embedding_cols) > 1:
        print(f"Warning: Multiple embedding columns found: {embedding_cols}. Using the first one.")
    return embedding_cols[0]

def parse_embedding(value):
    """Parse embedding value, handling different formats."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except:
            return value.split(',')
    elif isinstance(value, list):
        return value
    else:
        return [value] 

def load_and_process_csv(file_path):
    print(f"Loading file: {file_path}")
    df = pd.read_csv(file_path)
    embedding_col = find_embedding_column(df)
    
    df[embedding_col] = df[embedding_col].apply(parse_embedding)
    
    return df[['author', 'original_text', embedding_col]]

def consolidate_embeddings(file_paths, output_file):
    dfs = []
    for file_path in file_paths:
        df = load_and_process_csv(file_path)
        dfs.append(df)
    
    consolidated_df = dfs[0]
    for df in dfs[1:]:
        consolidated_df = pd.merge(consolidated_df, df, on=['author', 'original_text'], suffixes=('', f'_{file_paths.index(df)}'))
    
    print(f"Saving consolidated data to {output_file}")
    print(f"Final shape: {consolidated_df.shape}")
    print(f"Columns: {consolidated_df.columns.tolist()}")
    for col in consolidated_df.columns:
        if 'embedding' in col.lower():
            consolidated_df[col] = consolidated_df[col].apply(lambda x: np.array2string(np.array(x), separator=',', threshold=np.inf))
    
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
