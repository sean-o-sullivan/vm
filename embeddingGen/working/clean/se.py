import pandas as pd
import ast

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f"Error converting string to list: {e}")
        return s

def combine_texts_and_embeddings(embeddings_file, texts_file, output_file):
    print("Starting combine_texts_and_embeddings function")
    
    print(f"Loading embeddings from {embeddings_file}")
    embeddings_df = pd.read_csv(embeddings_file)
    print(f"Embeddings DataFrame shape: {embeddings_df.shape}")
    embeddings_df['embedding'] = embeddings_df['embedding'].apply(string_to_list)
    
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
