import pandas as pd

def load_texts(texts_csv):
    return pd.read_csv(texts_csv)

def load_embeddings(embeddings_csv):
    df = pd.read_csv(embeddings_csv)
    df = df.iloc[:, 1:]  # Disregard the first two columns
    #'author' column is in row 2, include it
    df['embeddings'] = df.iloc[:, 1:].apply(lambda row: row.tolist(), axis=1)
    df = df[['author', 'embeddings']]
    return df

if __name__ == "__main__":

    texts_df = load_texts('/home/aiadmin/Desktop/code/vm/embeddingGen/working/AGG_30.csv')
    embeddings_df = load_embeddings('/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/GG_30.csv')
    

    if 'author' not in texts_df.columns:
        raise ValueError("'author' column is missing from the texts DataFrame.")

    texts_author_counts = texts_df['author'].value_counts()
    embeddings_author_counts = embeddings_df['author'].value_counts()

    missing_in_texts = []
    missing_in_embeddings = []

    for author in embeddings_author_counts.index:
        if author not in texts_author_counts:
            missing_in_texts.append(author)
        elif embeddings_author_counts[author] != texts_author_counts[author]:
            missing_in_texts.append(author)

    for author in texts_author_counts.index:
        if author not in embeddings_author_counts:
            missing_in_embeddings.append(author)
        elif texts_author_counts[author] != embeddings_author_counts[author]:
            missing_in_embeddings.append(author)

    if missing_in_texts:
        print(f"Authors present in embeddings but missing in texts or have mismatched counts: {missing_in_texts}")
    if missing_in_embeddings:
        print(f"Authors present in texts but missing in embeddings or have mismatched counts: {missing_in_embeddings}")

    if len(texts_df) != len(embeddings_df):
        raise ValueError("The number of rows in the texts and embeddings DataFrames still do not match after checking for missing authors.")
    
    texts_df['embeddings'] = embeddings_df['embeddings']
    output_file = 'combined_texts_and_embeddings.csv'
    texts_df.to_csv(output_file, index=False)
    
    print(f"Combined DataFrame saved to {output_file}")
    print(texts_df.head())
