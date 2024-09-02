import pandas as pd

def load_original_data(combined_data_file):

    print(f"Loading the combined data from {combined_data_file}")
    combined_data_df = pd.read_csv(combined_data_file)
    print(f"Combined data loaded. Shape: {combined_data_df.shape}")
    return combined_data_df[['author', 'original_text', 'embedding']]

def load_csv(file_path, embedding_column):

    print(f"Loading CSV file from {file_path}")
    df = pd.read_csv(file_path)
    print(f"CSV file loaded. Shape: {df.shape}")
    return df[['author', embedding_column, 'original_text']]

def create_comprehensive_dataframe(combined_data_file, mimic_files, topic_files, output_file):

    print("Loading original data...")
    original_data_df = load_original_data(combined_data_file)
    print("Original data loaded.")
        print("Loading all embeddings CSVs...")
    mimic_embeddings = {name: load_csv(file, 'generated_mimicry_embedding') for name, file in mimic_files.items()}
    topic_embeddings = {name: load_csv(file, 'generated_text_embedding') for name, file in topic_files.items()}
    print("All embeddings CSVs loaded.")
    
    combined_data = []
    
    print("Creating the combined DataFrame...")
    for i, row in original_data_df.iterrows():
        author = row['author']
        original_text = row['original_text']
        original_embedding = row['embedding']

        print(f"Processing row {i + 1}/{len(original_data_df)} - Author: {author}, Original Text: {original_text}")

        combined_row = {
            'author': author,
            'original_text': original_text,
            'original_embedding': original_embedding
        }
        
        match_found = True
        for name, df in mimic_embeddings.items():
            match = df[(df['author'] == author) & (df['original_text'] == original_text)]
            if not match.empty:
                print(f"Match found in {name} for author {author}.")
                combined_row[f'embedding_{name}'] = match.iloc[0]['generated_mimicry_embedding']
            else:
                print(f"No match found in {name} for author {author}.")
                match_found = False
                break
        
        if match_found:
            for name, df in topic_embeddings.items():
                match = df[(df['author'] == author) & (df['original_text'] == original_text)]
                if not match.empty:
                    print(f"Match found in {name} for author {author}.")
                    combined_row[f'embedding_{name}'] = match.iloc[0]['generated_text_embedding']
                else:
                    print(f"No match found in {name} for author {author}.")
                    match_found = False
                    break
        
        if match_found:
            print(f"Adding row for author {author} to combined data.")
            combined_data.append(combined_row)
        else:
            print(f"Skipping row for author {author} due to missing matches.")
    
    final_df = pd.DataFrame(combined_data)
    
    print(f"Saving the final DataFrame to {output_file}")
    final_df.to_csv(output_file, index=False)
    print(f"Final DataFrame saved to {output_file}. Shape: {final_df.shape}")

    return final_df

if __name__ == "__main__":
    print("Starting main execution")
    
    combined_data_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/clean/combined_data.csv'
    
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
        combined_data_file, mimic_files, topic_files, "comprehensive_dataframe.csv"
    )

    print("Main execution is now! complete")
