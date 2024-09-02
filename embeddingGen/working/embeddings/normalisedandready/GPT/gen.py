import pandas as pd
import ast

df_a = pd.read_csv('AGG_csv_a_with_embeddings.csv')
df_b = pd.read_csv('Final-Triplets_G_30_|2|_VTL5_C3.csv')

df_s = pd.DataFrame(columns=['anchor_embedding', 'mimic_GPT3AGG_embedding', 'mimic_GPT4TAGG_embedding', 
                             'mimic_GPT4oAGG_embedding', 'topic_GPT3AGG_embedding', 
                             'topic_GPT4TAGG_embedding', 'topic_GPT4oAGG_embedding'])

for index, row in df_a.iterrows():

    embedding_g = row['embeddings']
    embedding_g = ast.literal_eval(embedding_g.strip())
    matches = df_b[df_b['positive_embedding'].apply(lambda x: ast.literal_eval(x) == embedding_g)]
    
    if not matches.empty:
        for _, match_row in matches.iterrows():
           anchor_embedding = match_row['anchor_embedding']
            
            new_row = {
                'anchor_embedding': anchor_embedding,
                'mimic_GPT3AGG_embedding': row['mimic_GPT3AGG_embedding'],
                'mimic_GPT4TAGG_embedding': row['mimic_GPT4TAGG_embedding'],
                'mimic_GPT4oAGG_embedding': row['mimic_GPT4oAGG_embedding'],
                'topic_GPT3AGG_embedding': row['topic_GPT3AGG_embedding'],
                'topic_GPT4TAGG_embedding': row['topic_GPT4TAGG_embedding'],
                'topic_GPT4oAGG_embedding': row['topic_GPT4oAGG_embedding']
            }
            
            df_s = df_s.append(new_row, ignore_index=True)
df_s.to_csv('output_S.csv', index=False)

print("Processing complete. Output saved to 'output_S.csv'.")
