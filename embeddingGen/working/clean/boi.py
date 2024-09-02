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

def combine_csvs(original, mimics, topics, output):
    original_df = load_csv(original)
    mimic_dfs = [load_csv(file) for file in mimics]
    topic_dfs = [load_csv(file) for file in topics]
    df = original_df[['author', 'embedding']].copy()
    df.columns = ['author', 'original_embedding']
    df['original_embedding'] = df['original_embedding'].apply(string_to_list)
    
    df['original_text'] = original_df['original_text']
    for i, mimic_df in enumerate(mimic_dfs):
        sampled_texts = mimic_df['original_text'].sample(len(df), replace=True, random_state=i).reset_index(drop=True)
        df['original_text'] = sampled_texts
    
    for i, file in enumerate(mimics):
        model_name = file.split('_')[3]  # Extracting the GPT model name from the filename
        df[f'mimic_{model_name}_embedding'] = mimic_dfs[i]['embedding'].apply(string_to_list)
    
    for i, file in enumerate(topics):
        model_name = file.split('_')[3]  
        df[f'topic_{model_name}_embedding'] = topic_dfs[i]['embedding'].apply(string_to_list)
    df.to_csv(output, index=False)

if __name__ == "__main__":
    combine_csvs('combined_data.csv',
                 ['normalized_mimicry_samples_GPT3ABB_30_embeddings.csv',
                  'normalized_mimicry_samples_GPT4TABB_30_embeddings.csv',
                  'normalized_mimicry_samples_GPT4oABB_30_embeddings.csv'],
                 ['normalized_topic_based_samples_GPT3ABB_30_embeddings.csv',
                  'normalized_topic_based_samples_GPT4TABB_30_embeddings.csv',
                  'normalized_topic_based_samples_GPT4oABB_30_embeddings.csv'],
                 'combined_embeddings.csv')
