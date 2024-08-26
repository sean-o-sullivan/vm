import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#plotting and scheming 
csv_file = 'corpus_statistics.csv'


df = pd.read_csv(csv_file)


plt.figure(figsize=(10, 6))
plt.plot(df['Quotations Rate per Character'], 'bo-', label='Quotation Density')
plt.title('Quotation Density Across All Books')
plt.xlabel('Book Index')
plt.ylabel('Quotation Density (per character)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(df['Parentheticals Rate per Character'], 'ro-', label='Parentheticals/Brackets Density')
plt.title('Parentheticals/Brackets Density Across All Books')
plt.xlabel('Book Index')
plt.ylabel('Parentheticals/Brackets Density (per character)')
plt.legend()
plt.grid(True)
plt.show()


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
plt.plot(df['Quotation Z-Score'], 'go-', label='Quotation Z-Score')
plt.title('Quotation Z-Scores Across All Books')
plt.xlabel('Book Index')
plt.ylabel('Z-Score')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(df['Parentheticals Z-Score'], 'mo-', label='Parentheticals Z-Score')
plt.title('Parentheticals/Brackets Z-Scores Across All Books')
plt.xlabel('Book Index')
plt.ylabel('Z-Score')
plt.legend()
plt.grid(True)
plt.show()
