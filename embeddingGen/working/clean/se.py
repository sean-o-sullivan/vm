import pandas as pd
import ast

def convert_row_to_list(row):
    return row.tolist()[1:]

def combine_texts_and_embeddings(embeddings_file, texts_file, output_file):
    print("Starting combine_texts_and_embeddings function")
    
    print(f"Loading embeddings from {embeddings_file}")
    embeddings_df = pd.read_csv(embeddings_file)
    print(f"Embeddings DataFrame shape: {embeddings_df.shape}")
    
    embeddings_df['embedding'] = embeddings_df.apply(convert_row_to_list, axis=1)
    embeddings_df = embeddings_df[['author', 'embedding']]
    
    print(f"Loading texts from {texts_file}")
    texts_df = pd.read_csv(texts_file)
    print(f"Texts DataFrame shape: {texts_df.shape}")

    print("Merging embeddings and texts")
    combined_df = pd.merge(embeddings_df, texts_df, on='author', how='inner')
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    final_df = combined_df[['author', 'original_text', 'embedding']]
    print(f"Saving combined DataFrame to {output_file}")
    final_df.to_csv(output_file, index=False)
    print("combine_texts_and_embeddings function completed")
    
    return final_df

embeddings_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/BB_30.csv'
texts_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/ABB_30.csv'
output_file = 'combined_texts_and_embeddings.csv'
result_df = combine_texts_and_embeddings(embeddings_file, texts_file, output_file)
print(result_df.head())
print("\nVerifying embedding list length:")
print(f"Number of items in the first embedding: {len(result_df['embedding'].iloc[0])}")
