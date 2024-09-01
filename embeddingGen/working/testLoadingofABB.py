import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def view_text_entries(input_file):
    df = pd.read_csv(input_file)
    if df.empty:
        logging.error(f"No entries found in {input_file}. Skipping.")
        return
    
    print(f"CSV Headers: {df.columns.tolist()}")
    print(f"Total entries: {len(df)}")
    entry_number = 1
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Viewing entries"):
        print(f"\nEntry {entry_number}:")
        print(f"Author: {row['author']}")
        print(f"Cleaned Text:\n{row['cleaned_text']}\n")
        input("Press Enter to view the next entry...")
        entry_number += 1

def main():
    view_text_entries('ABB.csv')
if __name__ == "__main__":
    main()
