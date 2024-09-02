from utils import find_matching_text, save_to_csv
import pandas as pd

def create_author_data(original_data_df):

    print("Creating author data dictionary")
    author_data = {row['author']: {'original_text': row['cleaned_text'], 
                                   'original_embedding': row['embedding']} 
                   for _, row in original_data_df.iterrows()}
    print(f"Number of authors processed: {len(author_data)}")
    return author_data

def process_adversarial_embeddings(author_data, adversarial_dfs):

    for adv_type, adv_df in adversarial_dfs.items():
        print(f"\nProcessing {adv_type} embeddings")
        print(f"Shape of {adv_type} dataframe: {adv_df.shape}")
        embedding_col = 'generated_mimicry_embedding' if 'mimic' in adv_type else 'generated_text_embedding'
        
        matches_found = 0
        for idx, adv_row in adv_df.iterrows():
            if idx % 100 == 0:
                print(f"Processing row {idx} of {adv_type}")
            
            match_found = False
            for author, data in author_data.items():
                if find_matching_text(adv_row['original_text'], data['original_text']):
                    author_data[author][adv_type] = adv_row[embedding_col]
                    matches_found += 1
                    match_found = True
                    break
            
            if not match_found:
                print(f"No match found for row {idx} in {adv_type}")
        
        print(f"Total matches found for {adv_type}: {matches_found}")
    
    return author_data

def create_comprehensive_df(author_data):
  
    print("Creating comprehensive dataframe")
    comprehensive_df = pd.DataFrame.from_dict(author_data, orient='index')
    comprehensive_df.reset_index(inplace=True)
    comprehensive_df.rename(columns={'index': 'author'}, inplace=True)
    
    expected_columns = ['author', 'original_text', 'original_embedding', 'gpt3_mimic', 'gpt3_raw', 
                        'gpt4t_mimic', 'gpt4t_raw', 'gpt4o_mimic', 'gpt4o_raw']
    
    for col in expected_columns:
        if col not in comprehensive_df.columns:
            print(f"Adding missing column: {col}")
            comprehensive_df[col] = None
    
    comprehensive_df = comprehensive_df[expected_columns]
    print(f"Comprehensive dataframe shape: {comprehensive_df.shape}")
    print(f"Columns: {comprehensive_df.columns.tolist()}")
    return comprehensive_df
