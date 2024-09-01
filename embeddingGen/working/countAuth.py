import pandas as pd

def count_unique_authors(csv_file):
    df = pd.read_csv(csv_file)
    unique_authors = df['author'].nunique()
    return unique_authors

if __name__ == "__main__":
#    csv_file = 'BAWE_texts.csv'
    csv_file = 'ABB_30.csv'
    unique_authors_count = count_unique_authors(csv_file)
    print(f"The number of unique authors in {csv_file} is: {unique_authors_count}")
