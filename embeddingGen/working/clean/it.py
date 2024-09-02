import pandas as pd

def load_original_data(embeddings_file, texts_file):

    print(f"Loading embeddings from {embeddings_file}")
    embeddings_df = pd.read_csv(embeddings_file)
    embedding_columns = embeddings_df.columns.difference(['embedding_id', 'author'])
    embeddings_df['embedding'] = embeddings_df[embedding_columns].apply(lambda row: row.tolist(), axis=1)
    embeddings_df = embeddings_df[['author', 'embedding']]
    
    print(f"Loading t. from {texts_file}")
    texts_df = pd.read_csv(texts_file)
    combined_df = pd.merge(embeddings_df, texts_df, on='author', how='inner')
    
    return combined_df[['author', 'cleaned_text', 'embedding']]

def load_csv(file_path, embedding_column):

    df = pd.read_csv(file_path)
    return df[['author', embedding_column]]

def create_comprehensive_dataframe(original_embeddings_file, original_texts_file,
                                   mimic_files, topic_files, output_file):

    original_data_df = load_original_data(original_embeddings_file, original_texts_file)
    
    print("Loading all embeddings CSVs")
    mimic_embeddings = {name: load_csv(file, 'generated_mimicry_embedding') for name, file in mimic_files.items()}
    topic_embeddings = {name: load_csv(file, 'generated_text_embedding') for name, file in topic_files.items()}
    
    min_length = min(
        len(original_data_df),
        *[len(df) for df in mimic_embeddings.values()],
        *[len(df) for df in topic_embeddings.values()]
    )
    
    combined_data = []
    
    print("Creating the combined DataFrame")
    for i in range(min_length):
        combined_row = {
            'author': original_data_df.loc[i, 'author'],
            'original_text': original_data_df.loc[i, 'cleaned_text'],
            'original_embedding': original_data_df.loc[i, 'embedding']
        }

        for name, df in mimic_embeddings.items():
            combined_row[f'embedding_{name}'] = df.loc[i, 'generated_mimicry_embedding']
        
        for name, df in topic_embeddings.items():
            combined_row[f'embedding_{name}'] = df.loc[i, 'generated_text_embedding']
        
        combined_data.append(combined_row)
    
    final_df = pd.DataFrame(combined_data)
    
    print(f"Saving the final DataFrame to {output_file}")
    final_df.to_csv(output_file, index=False)
    print(f"Final DataFrame saved to {output_file}")

    return final_df

if __name__ == "__main__":
    print("Starting main execution")
    original_embeddings_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/BB_30.csv'
    original_texts_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/ABB_30.csv'
    
    mimic_files = {
        'gpt3_mimic': '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT3ABB_30_embeddings.csv',
        'gpt4t_mimic': '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4TABB_30_embeddings.csv',
        'gpt4o_mimic': '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4oABB_30_embeddings.csv'
    }
    
    topic_files = {
        'gpt3_raw': '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT3ABB_30_embeddings.csv',
        'gpt4t_raw': '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4TABB_30_embeddings.csv',
        'gpt4o_raw': '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4oABB_30_embeddings.csv'
    }

    comprehensive_df = create_comprehensive_dataframe(
        original_embeddings_file, original_texts_file,
        mimic_files, topic_files, "comprehensive_dataframe.csv"
    )

    print("Main execution is now! complete")
