import pandas as pd
import numpy as np
import os
import ast

def normalize_future_csv(csv_path, stats_data_path, features_to_omit, output_dir):
    
    stats_data = pd.read_csv(stats_data_path, index_col=0)
    df = pd.read_csv(csv_path)
    if 'generated_mimicry_embedding' in df.columns:
        embedding_column = 'generated_mimicry_embedding'
    elif 'generated_text_embedding' in df.columns:
        embedding_column = 'generated_text_embedding'
    else:
        raise ValueError("No recognized embedding column found in the CSV. :/")

    def normalize_embedding(embedding_str):
        embedding = ast.literal_eval(embedding_str)
        normalized_embedding = []
        
        for i, value in enumerate(embedding):
            col = str(i)
            if col not in features_to_omit and col in stats_data.index:
                mean = stats_data.at[col, 'mean']
                std = stats_data.at[col, 'std']
                percentile_99_5 = stats_data.at[col, 'percentile_99.5']
                percentile_0_5 = stats_data.at[col, 'percentile_0.5']
                
                if std != 0:
                    z_score = (value - mean) / std
                    z_score_0_5 = (percentile_0_5 - mean) / std
                    z_score_99_5 = (percentile_99_5 - mean) / std
                    normalized_value = (z_score - z_score_0_5) / (z_score_99_5 - z_score_0_5)
                    normalized_value = np.clip(normalized_value, 0, 1)
                else:
                    normalized_value = 0.5
                
                normalized_embedding.append(normalized_value)
        
        return str(normalized_embedding)
    df[embedding_column] = df[embedding_column].apply(normalize_embedding)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"normalized_{os.path.basename(csv_path)}")
    df.to_csv(output_file_path, index=False)
    
    print(f"Normalized CSV saved to: {output_file_path}")

stats_data_path = '../embedding_stats.csv'
features_to_omit = [
    "ratio_of_sentence_initial_conjunctions",
    "detailed_conjunctions_usage_correlative",
    "normalized_assonance"
]
output_dir = 'normalized_adversarial_csvs'
adversarial_csvs = [
    '../mimicry_samples_GPT3ABB_30_embeddings.csv',
    '../mimicry_samples_GPT3AGG_30_embeddings.csv',
    '../mimicry_samples_GPT4oABB_30_embeddings.csv',
    '../mimicry_samples_GPT4oAGG_30_embeddings.csv',
    '../mimicry_samples_GPT4TABB_30_embeddings.csv',
    '../mimicry_samples_GPT4TAGG_30_embeddings.csv',
    '../topic_based_samples_GPT3ABB_30_embeddings.csv',
    '../topic_based_samples_GPT3AGG_30_embeddings.csv',
    '../topic_based_samples_GPT4oABB_30_embeddings.csv',
    '../topic_based_samples_GPT4oAGG_30_embeddings.csv',
    '../topic_based_samples_GPT4TABB_30_embeddings.csv',
    '../topic_based_samples_GPT4TAGG_30_embeddings.csv'
]
for csv_file in adversarial_csvs:
    normalize_future_csv(csv_file, stats_data_path, features_to_omit, output_dir)
