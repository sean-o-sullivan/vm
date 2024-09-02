import pandas as pd

def load_texts(filepath):
    return pd.read_csv(filepath)

def load_embeddings(filepath):
    df = pd.read_csv(filepath).iloc[:, 1:]  # Ignore first column
    df['embeddings'] = df.apply(lambda row: row.tolist(), axis=1)
    return df[['author', 'embeddings']]

def main():

    texts_df = load_texts('/home/aiadmin/Desktop/code/vm/embeddingGen/working/AGG_30.csv')
    embeddings_df = load_embeddings('/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/GG_30.csv')
    
    if 'author' not in texts_df.columns:
        raise ValueError("'author' column is missing in the texts DataFrame.")

    texts_authors, embeddings_authors = set(texts_df['author']), set(embeddings_df['author'])
    missing_in_texts = embeddings_authors - texts_authors
    missing_in_embeddings = texts_authors - embeddings_authors

    if missing_in_texts:
        print(f"Authors in embeddings but missing in texts: {missing_in_texts}")
    if missing_in_embeddings:
        print(f"Authors in texts but missing in embeddings: {missing_in_embeddings}")
    if len(texts_df) != len(embeddings_df):
        raise ValueError("Row counts differ between texts and embeddings DataFrames.")
    texts_df['embeddings'] = embeddings_df['embeddings']
    output_file = 'combined_texts_and_embeddings.csv'
    texts_df.to_csv(output_file, index=False)
    print(f"Combined DataFrame saved to {output_file}")
    print(texts_df.head())

if __name__ == "__main__":
    main()
