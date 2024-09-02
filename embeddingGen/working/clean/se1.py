import pandas as pd

def load_texts(texts_csv):
    return pd.read_csv(texts_csv)

def load_embeddings(embeddings_csv):
    df = pd.read_csv(embeddings_csv)
    df = df.iloc[:, 2:]  # Disregard the first two columns
    df['embeddings'] = df.apply(lambda row: row.tolist(), axis=1)
    df = df[['embeddings']]
    return df

if __name__ == "__main__":

    texts_df = load_texts('/home/aiadmin/Desktop/code/vm/embeddingGen/working/ABB_30.csv')
    embeddings_df = load_embeddings('/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/BB_30.csv')
    if 'author' not in texts_df.columns:
        raise ValueError("'author' column is missing from the texts DataFrame.")

    texts_authors = set(texts_df['author'])
    embeddings_authors = set(embeddings_df.index) 

    missing_in_texts = embeddings_authors - texts_authors
    missing_in_embeddings = texts_authors - embeddings_authors

    if missing_in_texts:
        print(f"Authors present in embeddings but missing in texts: {missing_in_texts}")
    if missing_in_embeddings:
        print(f"Authors present in texts but missing in embeddings: {missing_in_embeddings}")

    if len(texts_df) != len(embeddings_df):
        raise ValueError("The number of rows in the texts and embeddings DataFrames still do not match after checking for missing authors.")
    
    texts_df['embeddings'] = embeddings_df['embeddings']
    output_file = 'combined_texts_and_embeddings.csv'
    texts_df.to_csv(output_file, index=False)
    
    print(f"Combined DataFrame saved to {output_file}")
    print(texts_df.head())
