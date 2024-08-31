import csv
csv_file_path = 'authors_texts.csv'  
csv_file_path = 'BAWE_texts.csv'  

with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)

    for row in reader:
        author_id = row[0]
        text = row[1]
        print(f"Author ID: {author_id}")
        print(f"Text snippet: {text}...") 
        print("-" * 50)
        input(">>> press enter to continue")

print("Test is finally completed. See output to ensure the text is properly formatted.")
