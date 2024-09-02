import pandas as pd

text_file_path = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/ABB_30.csv'
embeddings_file_path = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/BB_30.csv'
output_file_path = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/clean/combined_texts_and_embeddings.csv'
text_df = pd.read_csv(text_file_path)
embeddings_df = pd.read_csv(embeddings_file_path)

embeddings_df = embeddings_df.drop(columns=['EmbeddingID'])
merged_df = pd.merge(text_df, embeddings_df, left_on='author', right_on='authorID')
merged_df['embedding'] = merged_df.iloc[:, 3:].values.tolist()
final_df = merged_df[['author', 'original_text', 'embedding']]

final_df.to_csv(output_file_path, index=False)
