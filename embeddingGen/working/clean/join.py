import pandas as pd

csv_a = pd.read_csv('AGG_combined_embeddings.csv') #gpt data
csv_b = pd.read_csv('AGG_combined_texts_and_embeddings.csv') # only 3 columns

csv_a['original_text'] = csv_a['original_text'].astype(str).str[:100]
csv_b['original_text'] = csv_b['original_text'].astype(str).str[:100]
filtered_csv_b = csv_b[csv_b['original_text'].isin(csv_a['original_text'])]
merged_df = pd.merge(csv_a, filtered_csv_b[['original_text', 'embeddings']], on='original_text', how='left')
merged_df.to_csv('csv_a_with_embeddings.csv', index=False)

print("CSV A with embeddings has been saved as 'csv_a_with_embeddings.csv'.")
