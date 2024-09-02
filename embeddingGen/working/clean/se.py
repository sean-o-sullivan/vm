import pandas as pd
import ast
import random

def load_csv(file):
    return pd.read_csv(file)

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except:
        return s

def combine_csvs(mimics, topics, output):

    mimic_dfs = [load_csv(file) for file in mimics]
    topic_dfs = [load_csv(file) for file in topics]
    df = mimic_dfs[0][['author']].copy()
    
    author_text_map = {}
    for author in df['author'].unique():
        source_df = random.choice(mimic_dfs + topic_dfs)
        author_text_map[author] = source_df[source_df['author'] == author]['original_text'].sample(n=1, random_state=42).values[0]

    df['original_text'] = df['author'].map(author_text_map)
    for i, file in enumerate(mimics):
        model_name = file.split('_')[3]  # Extracting GPT model name from the filename
        column_name = f'mimic_{model_name}_embedding'
        df[column_name] = mimic_dfs[i]['embedding'].apply(string_to_list)
    
    for i, file in enumerate(topics):
        model_name = file.split('_')[3]  
        column_name = f'topic_{model_name}_embedding'
        df[column_name] = topic_dfs[i]['embedding'].apply(string_to_list)
    
    df.to_csv(output, index=False)

if __name__ == "__main__":
    combine_csvs(['/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT3ABB_30_embeddings.csv',
                  '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4TABB_30_embeddings.csv',
                  '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4oABB_30_embeddings.csv'],
                 ['/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT3ABB_30_embeddings.csv',
                  '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4TABB_30_embeddings.csv',
                  '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4oABB_30_embeddings.csv'],
                 'combined_embeddings.csv')
