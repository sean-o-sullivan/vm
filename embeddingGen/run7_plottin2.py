import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#plotting more
csv_file = 'corpus_statistics.csv'

# Load the dataainto a pandas DataFrame
df = pd.read_csv(csv_file)


plt.figure(figsize=(10, 6))
plt.hist(df['Quotations Rate per Character'], bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Quotation Density Across All Books')
plt.xlabel('Quotation Density (per character)')
plt.ylabel('Number of Books')
plt.grid(True)
plt.savefig('histogram_quotation_density.png')
plt.close()


plt.figure(figsize=(10, 6))
plt.hist(df['Parentheticals Rate per Character'], bins=50, color='red', alpha=0.7)
plt.title('Histogram of Parentheticals/Brackets Density Across All Books')
plt.xlabel('Parentheticals/Brackets Density (per character)')
plt.ylabel('Number of Books')
plt.grid(True)
plt.savefig('histogram_parentheticals_density.png')
plt.close()


def calculate_z_scores(series):
    mean = series.mean()
    std = series.std()
    z_scores = []
    for value in tqdm(series, desc="Calculating Z-scores", unit="book"):
        z_score = (value - mean) / std if std != 0 else 0
        z_scores.append(z_score)
    return np.array(z_scores)


df['Quotation Z-Score'] = calculate_z_scores(df['Quotations Rate per Character'])
df['Parentheticals Z-Score'] = calculate_z_scores(df['Parentheticals Rate per Character'])


plt.figure(figsize=(10, 6))
plt.hist(df['Quotation Z-Score'], bins=50, color='green', alpha=0.7)
plt.title('Histogram of Quotation Z-Scores Across All Books')
plt.xlabel('Quotation Z-Score')
plt.ylabel('Number of Books')
plt.grid(True)
plt.savefig('histogram_quotation_z_scores.png')
plt.close()


plt.figure(figsize=(10, 6))
plt.hist(df['Parentheticals Z-Score'], bins=50, color='magenta', alpha=0.7)
plt.title('Histogram of Parentheticals/Brackets Z-Scores Across All Books')
plt.xlabel('Parentheticals Z-Score')
plt.ylabel('Number of Books')
plt.grid(True)
plt.savefig('histogram_parentheticals_z_scores.png')
plt.close()
