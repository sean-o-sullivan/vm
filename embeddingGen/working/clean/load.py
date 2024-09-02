import pandas as pd

def load(embeddings_file, texts_file, output_file):

    print(f"Loading embeddings from {embeddings_file}")
    embeddings_df = pd.read_csv(embeddings_file)
    print(f"Embeddings DF shape: {embeddings_df.shape}")
    
    embedding_columns = embeddings_df.columns.difference(['embedding_id', 'author'])
    embeddings_df['embedding'] = embeddings_df[embedding_columns].apply(lambda row: row.tolist(), axis=1)
    embeddings_df = embeddings_df[['author', 'embedding']]
    print(f"Processed Embeddings DF shape: {embeddings_df.shape}")
    print(f"Loading texts from {texts_file}")
    texts_df = pd.read_csv(texts_file)
    print(f"Texts DF shape: {texts_df.shape}")
    print("Merging embeddings and texts")
    combined_df = pd.merge(embeddings_df, texts_df, on='author', how='inner')
    print(f"Combined DF shape: {combined_df.shape}")

    final_df = combined_df[['author', 'cleaned_text', 'embedding']]
    print(f"Saving combined DataFrame to {output_file}")
    final_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    return final_df

original_embeddings_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/BB_30.csv'
original_texts_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/ABB_30.csv'
output_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/combined_data.csv'
original_data_df = load(original_embeddings_file, original_texts_file, output_file)
