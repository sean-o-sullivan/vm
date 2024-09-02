import pandas as pd

def combine_texts_and_embeddings(embeddings_file, texts_file, output_file):
    print("Starting combine_texts_and_embeddings function")
    
    print(f"Loading embeddings from {embeddings_file}")
    embeddings_df = pd.read_csv(embeddings_file)
    print(f"Embeddings DataFrame shape: {embeddings_df.shape}")
    embedding_columns = embeddings_df.columns[2:]
    embeddings_df['embedding'] = embeddings_df[embedding_columns].values.tolist()
    
    embeddings_df = embeddings_df[['author', 'embedding']]
    
    print(f"Loading texts from {texts_file}")
    texts_df = pd.read_csv(texts_file)
    print(f"Texts DataFrame shape: {texts_df.shape}")
    print("Combining embeddings and texts")
    combined_df = pd.merge(texts_df[['author', 'original_text']], embeddings_df, on='author', how='inner')
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    combined_df = combined_df[['author', 'original_text', 'embedding']]
    print(f"Saving combined DataFrame to {output_file}")
    combined_df.to_csv(output_file, index=False)
    print("combine_texts_and_embeddings function completed")
    
    return combined_df

embeddings_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/BB_30.csv'
texts_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/ABB_30.csv'
output_file = 'combined_texts_and_embeddings.csv'
result_df = combine_texts_and_embeddings(embeddings_file, texts_file, output_file)
print(result_df.head())
print("\nVerifying embedding list length:")
print(f"Number of items present in the first embedding: {len(result_df['embedding'].iloc[0])}")
