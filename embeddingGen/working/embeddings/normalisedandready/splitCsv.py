import pandas as pd
import os
import math

def find_file(pattern):
    command = f'find . -name "{pattern}"'
    result = os.popen(command).read().strip()
    return result if result else None

def split_csv(filepath, max_size_mb=95):
    base_path = os.path.splitext(filepath)[0]
    file_size = os.path.getsize(filepath)
    total_chunks = math.ceil(file_size / (max_size_mb * 1024 * 1024))
    df = pd.read_csv(filepath)
    rows_per_chunk = len(df) // total_chunks

    for i in range(total_chunks):
        start_idx = i * rows_per_chunk
        end_idx = None if i == total_chunks - 1 else (i + 1) * rows_per_chunk
       
        chunk = df.iloc[start_idx:end_idx]
        output_path = f"{base_path}_part{i+1}.csv"
        chunk.to_csv(output_path, index=False)
        print(f"Created {output_path} - Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

files_to_process = ["Final-Triplets_ters_|3|_VTL10_C2.csv"]
c5_file = find_file("*VTL10_C5.csv")
if c5_file:
    print(f"Found? C5 file at: {c5_file}")
    files_to_process.append(c5_file)
else:
    print("C5 file not found in directory tree")

for file in files_to_process:
    if os.path.exists(file):
        print(f"\nProcessing {file}")
        split_csv(file)
    else:
        print(f"File not found: {file}")