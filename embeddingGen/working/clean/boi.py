import pandas as pd
import ast
import random
import os

def load_csv(file):
    try:
        print(f"Loading file: {file}")
        df = pd.read_csv(file)
        print(f"Successfully loaded {file}. Columns: {df.columns}")
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
    print("Starting combine_csvs function")
    
    print("Loading mimic files:")
    mimic_dfs = [load_csv(file) for file in mimics]
    print("Loading topic files:")
    topic_dfs = [load_csv(file) for file in topics]
    print("Initializing DataFrame")
    df = mimic_dfs[0][['author', 'original_text']].copy()
    print(f"Initial DataFrame columns: {df.columns}")
    
    print("Processing mimic embeddings:")
    for i, file in enumerate(mimics):
        print(f"Processing mimic file: {file}")
        model_name = os.path.basename(file).split('_')[2]  # Adjust this index if necessary
        print(f"Extracted model name: {model_name}")
        column_name = f'mimic_{model_name}_embedding'
        print(f"Created column name: {column_name}")
        if 'embedding' in mimic_dfs[i].columns:
            print(f"Adding {column_name} to DataFrame")
            df[column_name] = mimic_dfs[i]['embedding'].apply(string_to_list)
        else:
            print(f"Missing 'embedding' column in {file}")
        print(f"Current DataFrame columns: {df.columns}")
    
    print("Processing topic embeddings:")
    for i, file in enumerate(topics):
        print(f"Processing topic file: {file}")
        model_name = os.path.basename(file).split('_')[3]  # Adjust this index if necessary
        print(f"Extracted model name: {model_name}")
        column_name = f'topic_{model_name}_embedding'
        print(f"Created column name: {column_name}")
        if 'embedding' in topic_dfs[i].columns:
            print(f"Adding {column_name} to DataFrame")
            df[column_name] = topic_dfs[i]['embedding'].apply(string_to_list)
        else:
            print(f"Missing 'embedding' column in {file}")
        print(f"Current DataFrame columns: {df.columns}")
    
    print(f"Final columns in the DataFrame: {df.columns}")
    print(f"Saving combined DataFrame to {output}")
    df.to_csv(output, index=False)
    print("Combine_csvs function completed")

if __name__ == "__main__":
    mimics = [
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT3ABB_30_embeddings.csv',
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4TABB_30_embeddings.csv',
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4oABB_30_embeddings.csv'
    ]
    topics = [
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT3ABB_30_embeddings.csv',
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4TABB_30_embeddings.csv',
        '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4oABB_30_embeddings.csv'
    ]
    combine_csvs(mimics, topics, 'combined_embeddings.csv')
    #hi
