import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


csv_file = 'corpus_statistics.csv'


df = pd.read_csv(csv_file)


num_bins = 50
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(df['Quotations Rate per Character'], bins=num_bins, color='blue', alpha=0.7)
plt.title('Histogram of Quotation Density Across All Books')
plt.xlabel('Quotation Density (per character)')
plt.ylabel('Number of Books')
plt.grid(True)
plt.show()


print("\nBooks per interval for Quotation Density:")
for i in range(len(bins)-1):
    print(f"Interval {bins[i]:.6f} - {bins[i+1]:.6f}: {int(counts[i])} books")


top_10_quotation = df.nlargest(10, 'Quotations Rate per Character')
print("\nTop 10 books with the highest Quotation Density:")
print(top_10_quotation[['Author', 'Book', 'Quotations Rate per Character']])


plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(df['Parentheticals Rate per Character'], bins=num_bins, color='red', alpha=0.7)
plt.title('Histogram of Parentheticals/Brackets Density Across All Books')
plt.xlabel('Parentheticals/Brackets Density (per character)')
plt.ylabel('Number of Books')
plt.grid(True)
plt.show()


print("\nBooks per interval for Parentheticals/Brackets Density:")
for i in range(len(bins)-1):
    print(f"Interval {bins[i]:.6f} - {bins[i+1]:.6f}: {int(counts[i])} books")


top_10_parentheticals = df.nlargest(10, 'Parentheticals Rate per Character')
print("\nTop 10 books with the highest Parentheticals/Brackets Density:")
print(top_10_parentheticals[['Author', 'Book', 'Parentheticals Rate per Character']])
