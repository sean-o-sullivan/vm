from load_data import load_all_data
from process_data import create_author_data, process_adversarial_embeddings, create_comprehensive_df
from utils import save_to_csv
import pandas as pd

def create_comprehensive_dataframe(original_embeddings_file, original_texts_file,
                                   gpt3_mimic_file, gpt3_raw_file,
                                   gpt4t_mimic_file, gpt4t_raw_file,
                                   gpt4o_mimic_file, gpt4o_raw_file,
                                   output_file):

    print("Starting comprehensive dataframe creation process")
    
    adversarial_files = {
        'gpt3_mimic': gpt3_mimic_file,
        'gpt3_raw': gpt3_raw_file,
        'gpt4t_mimic': gpt4t_mimic_file,
        'gpt4t_raw': gpt4t_raw_file,
        'gpt4o_mimic': gpt4o_mimic_file,
        'gpt4o_raw': gpt4o_raw_file
    }
    print("Loading all data")
    data = load_all_data(original_embeddings_file, original_texts_file, adversarial_files)
    print("Data loading complete")
    
    print("Creating author data dictionary")
    author_data = create_author_data(data['original'])
    save_to_csv(pd.DataFrame(author_data).T, 'author_data.csv')
    print("Author data dictionary created and saved")
    print("Processing adversarial embeddings")
    adversarial_dfs = {k: v for k, v in data.items() if k != 'original'}
    processed_author_data = process_adversarial_embeddings(author_data, adversarial_dfs)
    save_to_csv(pd.DataFrame(processed_author_data).T, 'processed_author_data.csv')
    print("Adversarial embeddings processed and saved")
    
    print("Creating final comprehensive dataframe")
    comprehensive_df = create_comprehensive_df(processed_author_data)
    
    print("Saving comprehensive dataframe to CSV")
    save_to_csv(comprehensive_df, output_file)
    print(f"Comprehensive dataframe saved to {output_file}")
    print(f"Final comprehensive dataframe shape: {comprehensive_df.shape}")
    
    print("\nSummary Statistics:")
    for column in comprehensive_df.columns:
        non_null_count = comprehensive_df[column].count()
        print(f"{column}: {non_null_count} non-null values")
    
    return comprehensive_df

if __name__ == "__main__":
    print("Starting main execution")
    original_embeddings_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/BB_30.csv'
    original_texts_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/ABB_30.csv'
    gpt3_mimic_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT3ABB_30_embeddings.csv'
    gpt3_raw_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT3ABB_30_embeddings.csv'
    gpt4t_mimic_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4TABB_30_embeddings.csv'
    gpt4t_raw_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4TABB_30_embeddings.csv'
    gpt4o_mimic_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_mimicry_samples_GPT4oABB_30_embeddings.csv'
    gpt4o_raw_file = '/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalized_adversarial_csvs/normalized_topic_based_samples_GPT4TABB_30_embeddings.csv'


    comprehensive_df = create_comprehensive_dataframe(
      original_embeddings_file, original_texts_file,
      gpt3_mimic_file, gpt3_raw_file,
      gpt4t_mimic_file, gpt4t_raw_file,
      gpt4o_mimic_file, gpt4o_raw_file,
      "comprehensive_dataframe.csv"
    )

    print("Main execution complete")
