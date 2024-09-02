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
    if len(texts_df) != len(embeddings_df):
        raise ValueError("The number of rows in the texts and embeddings DataFrames do not match. eek!.")
    
    texts_df['embeddings'] = embeddings_df['embeddings']
    print(texts_df.head())
