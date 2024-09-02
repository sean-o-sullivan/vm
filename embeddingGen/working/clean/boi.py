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
    df = mimic_dfs[0][['author', 'original_text', 'embedding']].copy()
    df.columns = ['author', 'original_text', 'original_embedding']
    df['original_embedding'] = df['original_embedding'].apply(string_to_list)
    
    for i in range(len(df)):
        source_df = random.choice(mimic_dfs + topic_dfs)
        df.at[i, 'original_text'] = source_df.at[i, 'original_text']
    
    for i, file in enumerate(mimics):
        model_name = file.split('_')[3]  # Extracting the GPT model name from the filename
        df[f'mimic_{model_name}_embedding'] = mimic_dfs[i]['embedding'].apply(string_to_list)
    
    for i, file in enumerate(topics):
        model_name = file.split('_')[3]  
        df[f'topic_{model_name}_embedding'] = topic_dfs[i]['embedding'].apply(string_to_list)
    df.to_csv(output, index=False)

if __name__ == "__main__":
    combine_csvs(['normalized_mimicry_samples_GPT3ABB_30_embeddings.csv',
                  'normalized_mimicry_samples_GPT4TABB_30_embeddings.csv',
                  'normalized_mimicry_samples_GPT4oABB_30_embeddings.csv'],
                 ['normalized_topic_based_samples_GPT3ABB_30_embeddings.csv',
                  'normalized_topic_based_samples_GPT4TABB_30_embeddings.csv',
                  'normalized_topic_based_samples_GPT4oABB_30_embeddings.csv'],
                 'combined_embeddings.csv')
