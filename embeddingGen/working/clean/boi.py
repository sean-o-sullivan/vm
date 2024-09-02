import pandas as pd
import ast
import random
import os

def load_csv(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f"Error converting string to list: {e}")
        return s

def combine_csvs(mimics, topics, output):

    mimic_dfs = [load_csv(file) for file in mimics]
    topic_dfs = [load_csv(file) for file in topics]
    df = mimic_dfs[0][['author', 'original_text']].copy()
    
    for i in range(len(df)):
        source_df = random.choice(mimic_dfs + topic_dfs)
        df.at[i, 'original_text'] = source_df.at[i, 'original_text']
    
    for i, file in enumerate(mimics):
        model_name = os.path.basename(file).split('_')[2]  # Extracting GPT model name from the filename
        column_name = f'mimic_{model_name}_embedding'
        if 'embedding' in mimic_dfs[i].columns:
            df[column_name] = mimic_dfs[i]['embedding'].apply(string_to_list)
        else:
            print(f"Missing 'embedding' column in {file}")
    
    for i, file in enumerate(topics):
        model_name = os.path.basename(file).split('_')[3]  # Extracting GPT model name from the filename
        column_name = f'topic_{model_name}_embedding'
        if 'embedding' in topic_dfs[i].columns:
            df[column_name] = topic_dfs[i]['embedding'].apply(string_to_list)
        else:
            print(f"Missing 'embedding' column in {file}")
        print(f"Final columns in the DataFrame: {df.columns}")
    df.to_csv(output, index=False)

if __name__ == "__main__":
    combine_csvs(['/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT3ABB_30_embeddings.csv',
                  '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4TABB_30_embeddings.csv',
                  '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4oABB_30_embeddings.csv'],
                 ['/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT3ABB_30_embeddings.csv',
                  '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4TABB_30_embeddings.csv',
                  '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4oABB_30_embeddings.csv'],
                 'combined_embeddings.csv')
