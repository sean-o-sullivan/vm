import pandas as pd

def load_texts(texts_csv):
    return pd.read_csv(texts_csv)

def load_embeddings(embeddings_csv):
    df = pd.read_csv(embeddings_csv)
    df = df.iloc[:, 1:]  # Disregard the first column
    df['embeddings'] = df.iloc[:, 1:].apply(lambda row: row.tolist(), axis=1)
    df = df[['author', 'embeddings']]
    return df

if __name__ == "__main__":

    texts_df = load_texts('/home/aiadmin/Desktop/code/vm/embeddingGen/working/AGG_30.csv')
    embeddings_df = load_embeddings('/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/GG_30.csv')
    
    if 'author' not in texts_df.columns:
        raise ValueError("'author' column is missing from the texts DataFrame.")

    texts_author_counts = texts_df['author'].value_counts()
    print(f"\nTexts Author Counts:\n{texts_author_counts}")
    
    embeddings_author_counts = embeddings_df['author'].value_counts()
    print(f"\nEmbeddings Author Counts:\n{embeddings_author_counts}")

    author_with_missing_occurrence = None
    for author in texts_author_counts.index:
        if texts_author_counts[author] != embeddings_author_counts.get(author, 0):
            difference = texts_author_counts[author] - embeddings_author_counts.get(author, 0)
            if abs(difference) == 1:
                author_with_missing_occurrence = author
                break

    if author_with_missing_occurrence:
        print(f"Author with a missing occurrence: {author_with_missing_occurrence}")
    else:
        print("No author with a missing occurrence was found.")

    if len(texts_df) != len(embeddings_df):
        raise ValueError("The number of rows in the texts and embeddings DataFrames do not match after checking for missing occurrences.")

    texts_df['embeddings'] = embeddings_df['embeddings']

    output_file = 'combined_texts_and_embeddings.csv'
    texts_df = texts_df.drop(texts_df.columns[0], axis=1)  # Drop the first column
    texts_df.to_csv(output_file, index=False)
    
    print(f"Combined DataFrame saved to {output_file}")
    print(texts_df.head())
