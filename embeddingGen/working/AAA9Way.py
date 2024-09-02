import pandas as pd
import numpy as np
import ast
import os

print("Script execution started")

def load_embeddings(embeddings_file, texts_file):
    print(f"Loading embeddings from {embeddings_file}")
    embeddings_df = pd.read_csv(embeddings_file)
    print(f"Embeddings DF shape: {embeddings_df.shape}")
    print(f"Embeddings DF head: {embeddings_df.head()}")
    
    print(f"Loading texts from {texts_file}")
    texts_df = pd.read_csv(texts_file)
    print(f"Texts DF shape: {texts_df.shape}")
    print(f"Texts DF head: {texts_df.head()}")
    
    print("Merging embeddings and texts")
    combined_df = pd.merge(embeddings_df, texts_file=texts_df, on='author', how='inner')
    print(f"Combined shape: {combined_df.shape}")
    
    embedding_columns = [col for col in combined_df.columns if col not in ['embedding_id', 'author', 'cleaned_text']]
    combined_df['embedding'] = combined_df[embedding_columns].values.tolist()
    
    return combined_df[['author', 'cleaned_text', 'embedding']]

def find_matching_text(original_text, adversarial_text, char_limit=500):

    original_substr = original_text[:char_limit]
    return original_substr in adversarial_text

def load_adversarial_embeddings(file_path, embedding_column):

    print(f"Loading adversarial embeddings from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Adversarial embeddings shape: {df.shape}")
    df[embedding_column] = df[embedding_column].apply(ast.literal_eval)
    return df

def create_comprehensive_dataframe(original_embeddings_file, original_texts_file, 
                                   gpt3_mimic_file, gpt3_raw_file,
                                   gpt4t_mimic_file, gpt4t_raw_file,
                                   gpt4o_mimic_file, gpt4o_raw_file,
                                   output_file):

    print("Creating comprehensive dataframe")
    
    # Load original embeddings and texts
    original_data_df = load_embeddings(original_embeddings_file, original_texts_file)
    print(f"Original data shape: {original_data_df.shape}")

    # Load all adversarial embeddings
    gpt3_mimic_df = load_adversarial_embeddings(gpt3_mimic_file, 'generated_mimicry_embedding')
    gpt3_raw_df = load_adversarial_embeddings(gpt3_raw_file, 'generated_text_embedding')
    gpt4t_mimic_df = load_adversarial_embeddings(gpt4t_mimic_file, 'generated_mimicry_embedding')
    gpt4t_raw_df = load_adversarial_embeddings(gpt4t_raw_file, 'generated_text_embedding')
    gpt4o_mimic_df = load_adversarial_embeddings(gpt4o_mimic_file, 'generated_mimicry_embedding')
    gpt4o_raw_df = load_adversarial_embeddings(gpt4o_raw_file, 'generated_text_embedding')

    comprehensive_data = []

    for idx, orig_row in original_data_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}")
        
        author = orig_row['author']
        original_text = orig_row['cleaned_text']
        original_embedding = orig_row['embedding']

        row_data = {
            'author': author,
            'original_text': original_text,
            'original_embedding': original_embedding
        }

        # Find that ridiculous matching adversarial embeddings
        for df, embedding_type in [
            (gpt3_mimic_df, 'gpt3_mimic'), (gpt3_raw_df, 'gpt3_raw'),
            (gpt4t_mimic_df, 'gpt4t_mimic'), (gpt4t_raw_df, 'gpt4t_raw'),
            (gpt4o_mimic_df, 'gpt4o_mimic'), (gpt4o_raw_df, 'gpt4o_raw')
        ]:
            matching_row = df[df['original_text'].apply(lambda x: find_matching_text(original_text, x))]
            
            if not matching_row.empty:
                embedding_col = 'generated_mimicry_embedding' if 'mimic' in embedding_type else 'generated_text_embedding'
                row_data[embedding_type] = matching_row.iloc[0][embedding_col]
                print(f"Match found for {embedding_type} - Author: {author}")
            else:
                print(f"Warning: No matching {embedding_type} embedding found for author {author}")
                row_data[embedding_type] = None

        comprehensive_data.append(row_data)
        
    comprehensive_df = pd.DataFrame(comprehensive_data)
    comprehensive_df.to_csv(output_file, index=False)
    print(f"Comprehensive dataframe saved to {output_file}")
    print(f"Final comprehensive dataframe shape: {comprehensive_df.shape}")
    
    return comprehensive_df

# Usage
print("Setting up file paths")

#we shall do ABB first
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
    original_embeddings_file, original_texts_file,
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
