import pandas as pd
import numpy as np
import ast

def find_matching_text(text1, text2, threshold=0.9):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    similarity = len(intersection) / max(len(words1), len(words2))
    print(f"Similarity between texts: {similarity:.2f}")
    return similarity >= threshold

def save_to_csv(df, filename):

    print(f"Saving dataframe to {filename}")
    print(f"Dataframe shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    for col in df.columns:
        if 'embedding' in col.lower():
            df[col] = df[col].apply(lambda x: np.array2string(x, separator=',', threshold=np.inf) if isinstance(x, np.ndarray) else x)

    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def load_from_csv(filename):

    print(f"Loading dataframe from {filename}")
    df = pd.read_csv(filename)
    print(f"Dataframe shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    for col in df.columns:
        if 'embedding' in col.lower():
            df[col] = df[col].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

    return df
