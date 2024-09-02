import pandas as pd
import ast
from tqdm import tqdm
import time

print("Script started. Importing necessary libraries...")

print("Reading File A...")
df_a = pd.read_csv('/home/aiadmin/Desktop/code/vm/embeddingGen/working/clean/AGG_csv_a_with_embeddings.csv')
print(f"File A loaded. Shape: {df_a.shape}")
print("Reading File B...")
df_b = pd.read_csv('/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/Final-Triplets_G_30_|2|_VTL5_C3.csv')
print(f"File B loaded. Shape: {df_b.shape}")

print("Initializing output DataFrame S...")
df_s = pd.DataFrame(columns=['anchor_embedding', 'mimic_GPT3AGG_embedding', 'mimic_GPT4TAGG_embedding', 
                             'mimic_GPT4oAGG_embedding', 'topic_GPT3AGG_embedding', 
                             'topic_GPT4TAGG_embedding', 'topic_GPT4oAGG_embedding'])

print("Starting to process rows from File A...")
total_matches = 0
start_time = time.time()

for index, row in tqdm(df_a.iterrows(), total=df_a.shape[0], desc="Processing rows"):

    embedding_g = row['embeddings']
    embedding_g = ast.literal_eval(embedding_g.strip())
    matches = df_b[df_b['positive_embedding'].apply(lambda x: ast.literal_eval(x) == embedding_g)]
    
    if not matches.empty:
        total_matches += len(matches)
        for _, match_row in matches.iterrows():
           anchor_embedding = match_row['anchor_embedding']
            
            new_row = pd.DataFrame({
                'anchor_embedding': [anchor_embedding],
                'mimic_GPT3AGG_embedding': [row['mimic_GPT3AGG_embedding']],
                'mimic_GPT4TAGG_embedding': [row['mimic_GPT4TAGG_embedding']],
                'mimic_GPT4oAGG_embedding': [row['mimic_GPT4oAGG_embedding']],
                'topic_GPT3AGG_embedding': [row['topic_GPT3AGG_embedding']],
                'topic_GPT4TAGG_embedding': [row['topic_GPT4TAGG_embedding']],
                'topic_GPT4oAGG_embedding': [row['topic_GPT4oAGG_embedding']]
            })
            
            df_s = pd.concat([df_s, new_row], ignore_index=True)

    if (index + 1) % 1000 == 0:
        elapsed_time = time.time() - start_time
        print(f"\nProcessed {index + 1} rows. Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Current matches found: {total_matches}")
        print(f"Current size of output DataFrame S: {df_s.shape}")
print("\nProcessing is now complete. Saving output to CSV...")
output_path = 'output_S.csv'
df_s.to_csv(output_path, index=False)

end_time = time.time()
total_time = end_time - start_time

print(f"Output saved to: {output_path}")
print(f"Total rows processed: {df_a.shape[0]}")
print(f"Total matches found: {total_matches}")
print(f"Final size of output DataFrame S: {df_s.shape}")
print(f"Total processing time: {total_time:.2f} seconds")
