import pandas as pd

csv_a = pd.read_csv('path_to_csv_a.csv')
filtered_csv_b = pd.read_csv('filtered_csv_b.csv')
csv_a['original_text'] = csv_a['original_text'].astype(str).str[:100]
filtered_csv_b['original_text'] = filtered_csv_b['original_text'].astype(str).str[:100]
merged_df = pd.merge(csv_a, filtered_csv_b[['original_text', 'embedding']], on='original_text', how='left')
merged_df.to_csv('csv_a_with_embeddings.csv', index=False)

print("CSV A with embeddings has been saved as 'csv_a_with_embeddings.csv'.")
