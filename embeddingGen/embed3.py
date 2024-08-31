import pandas as pd
import logging
from tqdm import tqdm
from embedding2 import generateEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_entry(row, embedding_columns):
    author = row['author']
    book_name = row['book']
    sample_id = row['sample_id']
    processed_sample = row['processed_sample']
#    input(f"the processed sample is very much: {processed_sample}")
    
    # Remove the custom delimiters from the processed_sample
    processed_sample = processed_sample.replace("#/#\\#|||#/#\\#|||#/#\\#", "")
    
    print(f"Processing sample_id: {sample_id}")
    print(f"Processed sample (first 100 chars): {processed_sample[:100]}")

    try:
        
        embedding = generateEmbedding(processed_sample)

        new_row = {
            'author': author,
            'book': book_name,
            'sample_id': sample_id
        }
        new_row.update(embedding)  # Add all key-value pairs from the embedding dictionary

        return pd.Series(new_row)
    except Exception as e:
        logging.error(f"Error processing sample_id {sample_id}: {str(e)}")
        return None

def main():
    input_csv = '/home/aiadmin/Desktop/code/vm/embeddingGen/Thursday/results_10KSample.csv'
    output_file = 'output_embeddings_10KSample.csv'

    df = pd.read_csv(input_csv)
    
    if df.empty:
        logging.error("No entries found in results.csv. Exiting.")
        return

    print(f"CSV Headers: {df.columns.tolist()}")
    print(f"Total entries: {len(df)}")

    sample_text = """
    We will then cross 17th Street and examine several buildings along 17th Street as we walk north towards Pennsylvania Avenue. The total distance is about three-fourths of a kilometer (half a mile). Capital Gatehouse Site 7 [Illustration: The Capitol Gatehouse, now located at 17th Street and Constitution Avenue, is made of the same sandstone used in the White House and the center part of the Capitol, but it was left unpainted. Deterioration of this stone is due to the clay it contains, not to the effects of acid rain.]
    """
    sample_embedding = generateEmbedding(sample_text)
    embedding_columns = list(sample_embedding.keys())

    tqdm.pandas(desc="Processing entries")
    result_df = df.progress_apply(lambda row: process_entry(row, embedding_columns), axis=1)

    result_df = result_df.dropna()
    result_df.to_csv(output_file, index=False)

    logging.info(f"Processing completed. Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()
