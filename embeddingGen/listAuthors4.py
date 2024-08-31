import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    input_csv = '/home/aiadmin/Desktop/code/vm/embeddingGen/Thursday/results_10KSample.csv'
    df = pd.read_csv(input_csv)
    
    if df.empty:
        logging.error("No entries found in results.csv. Exiting.")
        return

    unique_authors = df['author'].nunique()
    logging.info(f"Number of unique authors: {unique_authors}")

if __name__ == "__main__":
    main()
