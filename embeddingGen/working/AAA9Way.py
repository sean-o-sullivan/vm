import pandas as pd
import numpy as np
import ast
import os

print("Script execution started")

def load_embeddings(embeddings_file, texts_file):

    print(f"Loading embeddings from {embeddings_file}")
    embeddings_df = pd.read_csv(embeddings_file)
    print(f"Embeddings DF shape: {embeddings_df.shape}")

    embedding_columns = embeddings_df.columns.difference(['embedding_id', 'author'])
    embeddings_df['embedding'] = embeddings_df[embedding_columns].apply(lambda row: row.tolist(), axis=1)
    embeddings_df = embeddings_df[['author', 'embedding']]
    print(f"Processed Embeddings DF shape: {embeddings_df.shape}")

    print(f"Loading texts from {texts_file}")
    texts_df = pd.read_csv(texts_file)
    print(f"Texts DF shape: {texts_df.shape}")

    print("Merging embeddings and texts")
    combined_df = pd.merge(embeddings_df, texts_df, on='author', how='inner')
    print(f"Combined DF shape: {combined_df.shape}")

    # The final DataFrame will have columns: 'author', 'cleaned_text', 'embedding'
    return combined_df[['author', 'cleaned_text', 'embedding']]

def find_matching_text(adversarial_original_text, original_text, char_limit=100):

    adversarial_substr = adversarial_original_text[:char_limit]
    original_substr = original_text[:char_limit]

    original_substr = original_substr[3:92]
    adversarial_substr = adversarial_substr[1:90]

    return adversarial_substr == original_substr

def load_adversarial_embeddings(file_path, embedding_column):

    print(f"Loading adversarial embeddings from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Adversarial embeddings shape: {df.shape}")
    df[embedding_column] = df[embedding_column].apply(ast.literal_eval)
    return df



def create_comprehensive_dataframe(original_texts_file, original_embeddings_file,
                                   gpt3_mimic_file, gpt3_raw_file,
                                   gpt4t_mimic_file, gpt4t_raw_file,
                                   gpt4o_mimic_file, gpt4o_raw_file,
                                   output_file):

    print("Creating comprehensive dataframe")
    
    #  original embeddings and texts
    original_data_df = load_embeddings(original_embeddings_file, original_texts_file)
    print(f"Original data shape: {original_data_df.shape}")

    # Create a dictionary to store all embeddings for each author
    author_data = {row['author']: {'original_text': row['cleaned_text'], 
                                   'original_embedding': row['embedding']} 
                   for _, row in original_data_df.iterrows()}

    # Load all adversarial embeddings
    adversarial_dfs = {
        'gpt3_mimic': load_adversarial_embeddings(gpt3_mimic_file, 'generated_mimicry_embedding'),
        'gpt3_raw': load_adversarial_embeddings(gpt3_raw_file, 'generated_text_embedding'),
        'gpt4t_mimic': load_adversarial_embeddings(gpt4t_mimic_file, 'generated_mimicry_embedding'),
        'gpt4t_raw': load_adversarial_embeddings(gpt4t_raw_file, 'generated_text_embedding'),
        'gpt4o_mimic': load_adversarial_embeddings(gpt4o_mimic_file, 'generated_mimicry_embedding'),
        'gpt4o_raw': load_adversarial_embeddings(gpt4o_raw_file, 'generated_text_embedding')
    }

    for adv_type, adv_df in adversarial_dfs.items():
        print(f"\nProcessing {adv_type} embeddings")
        embedding_col = 'generated_mimicry_embedding' if 'mimic' in adv_type else 'generated_text_embedding'
        
        for idx, adv_row in adv_df.iterrows():
            if idx % 100 == 0:
                print(f"Processing row {idx} of {adv_type}")
            
            # Find matching original text
            for author, data in author_data.items():
                if find_matching_text(adv_row['original_text'], data['original_text']):
                    author_data[author][adv_type] = adv_row[embedding_col]
                    break

    comprehensive_df = pd.DataFrame.from_dict(author_data, orient='index')
    comprehensive_df.reset_index(inplace=True)
    comprehensive_df.rename(columns={'index': 'author'}, inplace=True)

    # Ensure all expected columns are present
    expected_columns = ['author', 'original_text', 'original_embedding', 'gpt3_mimic', 'gpt3_raw', 
                        'gpt4t_mimic', 'gpt4t_raw', 'gpt4o_mimic', 'gpt4o_raw']
    for col in expected_columns:
        if col not in comprehensive_df.columns:
            comprehensive_df[col] = None

    # Reorder columns
    comprehensive_df = comprehensive_df[expected_columns]

    # Save comprehensive dataframe
    print("Saving comprehensive dataframe to CSV")
    comprehensive_df.to_csv(output_file, index=False)
    print(f"Comprehensive dataframe saved to {output_file}")
    print(f"Final comprehensive dataframe shape: {comprehensive_df.shape}")
    
    return comprehensive_df



# Usage
print("Setting up file paths")

#  paths
original_embeddings_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/BB_30.csv'
original_texts_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/ABB_30.csv'
gpt3_mimic_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT3ABB_30_embeddings.csv'
gpt3_raw_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT3ABB_30_embeddings.csv'
gpt4t_mimic_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4TABB_30_embeddings.csv'
gpt4t_raw_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4TABB_30_embeddings.csv'
gpt4o_mimic_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4oABB_30_embeddings.csv'
gpt4o_raw_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4TABB_30_embeddings.csv'

output_file = 'comprehensive_dataframe.csv'

print("Calling create_comprehensive_dataframe function")
comprehensive_df = create_comprehensive_dataframe(
    original_texts_file, original_embeddings_file,
    gpt3_mimic_file, gpt3_raw_file,
    gpt4t_mimic_file, gpt4t_raw_file,
    gpt4o_mimic_file, gpt4o_raw_file,
    output_file
)

print("Displaying sample of the comprehensive dataframe")
print(comprehensive_df.head())

print("Checking for missing adversarial embeddings")
for embedding_type in ['gpt3_mimic', 'gpt3_raw', 'gpt4t_mimic', 'gpt4t_raw', 'gpt4o_mimic', 'gpt4o_raw']:
    missing_count = comprehensive_df[embedding_type].isna().sum()
    print(f"Number of missing {embedding_type} embeddings: {missing_count}")
print("Script execution completed")
