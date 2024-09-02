import pandas as pd
import ast

def load_csv(file_path):
    print(f"Loading CSV file from {file_path}")
    df = pd.read_csv(file_path)
    return df

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except:
        return s

def combine_csvs(original_file, mimic_files, topic_files, output_file):

    original_df = load_csv(original_file)
    mimic_dfs = [load_csv(file) for file in mimic_files]
    topic_dfs = [load_csv(file) for file in topic_files]
    combined_df = original_df[['author', 'original_text', 'embedding']]
    combined_df.columns = ['author', 'original_text', 'original_embedding']
    combined_df['original_embedding'] = combined_df['original_embedding'].apply(string_to_list)

    for i, df in enumerate(mimic_dfs):
        column_name = f'mimic_embedding_{i+1}'
        combined_df[column_name] = df['generated_mimicry_embedding'].apply(string_to_list)

    for i, df in enumerate(topic_dfs):
        column_name = f'topic_embedding_{i+1}'
        combined_df[column_name] = df['generated_text_embedding'].apply(string_to_list)

    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

if __name__ == "__main__":
    original_file = 'combined_data.csv'
    mimic_files = [
        'normalized_mimicry_samples_GPT3ABB_30_embeddings.csv',
        'normalized_mimicry_samples_GPT4TABB_30_embeddings.csv',
        'normalized_mimicry_samples_GPT4oABB_30_embeddings.csv'
    ]
    topic_files = [
        'normalized_topic_based_samples_GPT3ABB_30_embeddings.csv',
        'normalized_topic_based_samples_GPT4TABB_30_embeddings.csv',
        'normalized_topic_based_samples_GPT4oABB_30_embeddings.csv'
    ]
    output_file = 'combined_embeddings.csv'

    combine_csvs(original_file, mimic_files, topic_files, output_file)
