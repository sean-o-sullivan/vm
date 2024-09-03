import pandas as pd
import logging
from tqdm import tqdm
from embedding2 import generateEmbedding

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_entry(row, embedding_columns):
    author = row['author']
    text= row['text']
#    input(f"the processed sample is very much: {processed_sample}")
    
    
    print(f"Processed sample (first 100 chars): {text[:100]}")

    try:
        # Generate embedding
        embedding = generateEmbedding(text)

        # Create a new row with the embedding data
        new_row = {
            'author': author,
            'book': text[:100],
        }
        new_row.update(embedding)  # Add all key-value pairs from the embedding dictionary

        return pd.Series(new_row)
    except Exception as e:
        logging.error(f"Error processing sample_id {text[:100]}: {str(e)}")
        return None

def main():
    input_csv = '/home/aiadmin/Downloads/cleaned_reuters_corpus.csv'
    output_file = '/home/aiadmin/Downloads/output_embeddings_Reuters.csv'

    # Load the results CSV
    df = pd.read_csv(input_csv)
    
    if df.empty:
        logging.error("No entries found in results.csv. Exiting.")
        return

    print(f"CSV Headers: {df.columns.tolist()}")
    print(f"Total entries: {len(df)}")

    # Get the structure of the embedding dictionary using a sample entry
    sample_text = """
    We will then cross 17th Street and examine several buildings along 17th Street as we walk north towards Pennsylvania Avenue. The total distance is about three-fourths of a kilometer (half a mile). Capital Gatehouse Site 7 [Illustration: The Capitol Gatehouse, now located at 17th Street and Constitution Avenue, is made of the same sandstone used in the White House and the center part of the Capitol, but it was left unpainted. Deterioration of this stone is due to the clay it contains, not to the effects of acid rain.]
    """
    sample_embedding = generateEmbedding(sample_text)
    embedding_columns = list(sample_embedding.keys())

    # Process each entry and generate embeddings
    tqdm.pandas(desc="Processing entries")
    result_df = df.progress_apply(lambda row: process_entry(row, embedding_columns), axis=1)

    # Remove any rows that returned None due to errors
    result_df = result_df.dropna()

    # Save the result to the output CSV
    result_df.to_csv(output_file, index=False)

    logging.info(f"Processing completed. Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()
