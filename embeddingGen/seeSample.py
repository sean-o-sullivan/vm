import pandas as pd

def view_samples(file_path):
    
    df = pd.read_csv(file_path)
    if df.empty:
        print("No entries found in the file. Exiting.")
        return

    print(f"Total samples in file: {len(df)}")
    for _, row in df.iterrows():
        print(f"The Author: {row['author']}")
        print(f"Book is: {row['book']}")
        print(f"Sample ID: {row['sample_id']}")
        print(f"Processed final Sample:\n{row['processed_sample']}")
        input("Press Enter to see the next sample...")  
        print("\n" + "-"*50 + "\n")  

if __name__ == "__main__":
    output_file = 'output_raw_texts_with_delimiters.csv'
    view_samples(output_file)
