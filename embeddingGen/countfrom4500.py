import pandas as pd
file_path = '/home/aiadmin/Desktop/code/vm/embeddingGen/suitable_books_for_authorship_verification.csv'  # Please provide the correct path if necessary
try:
    df = pd.read_csv(file_path)
    unique_authors_count = df['Author'].nunique()
    print(unique_authors_count)
except Exception as e:
    print(str(e))