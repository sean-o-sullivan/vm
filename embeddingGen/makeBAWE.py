import os
import csv


directory_path = '/home/aiadmin/Downloads/download/CORPUS_TXT'  
output_csv = 'BAWE_texts.csv'

with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['author', 'text'])
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            author_id = filename.split('.')[0][:-1]
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
            enclosed_text = f'"""{text}"""'            
            writer.writerow([author_id, enclosed_text])


print(f"CSV file '{output_csv}' has been created successfully.")
