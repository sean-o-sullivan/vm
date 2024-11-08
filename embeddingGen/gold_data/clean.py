import pandas as pd
import re
from bs4 import BeautifulSoup
import unicodedata
from tqdm import tqdm

#whole lot of defitions from the internet
def clean_text(text):
    
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'\((?:\w+,?\s+(?:et al\.)?,?\s+)?(?:19|20)\d{2}[a-z]?(?::\d+(?:-\d+)?)?(?:\s+and\s+(?:\w+,?\s+(?:et al\.)?,?\s+)?(?:19|20)\d{2}[a-z]?(?::\d+(?:-\d+)?)?)*\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'(?m)^\s*[\|+].*[\|+]\s*$', '', text)
    text = re.sub(r'(?m)^\s*[-+]+\s*$', '', text)
    text = re.sub(r'(?m)^\s*[a-zA-Z0-9]+\s*[-+*/^()]+.*$', '', text)
    text = re.sub(r'(?m)^\s*[∑∫∏∂∇Δ].*$', '', text)
    text = re.sub(r'[±∓×÷∙∘·°∂∇∆∑∏∫√∛∜∝∞≈≠≡≤≥≪≫⊂⊃⊄⊅⊆⊇⊈⊉⊊⊋∈∉∋∌∍∎∏∐∑−]', '', text)
    text = re.sub(r'\b(\d+(?:\s+\d+)+)\b', '', text)
    text = re.sub(r'---+', '--', text)
    text = re.sub(r'[—–]', '-', text)
    text = re.sub(r'[""''""‹›«»]', "'", text)
    text = re.sub(r'[''´`]', "'", text)
    text = re.sub(r'[•◦▪▫▸▹►▻➤➢◆◇○●]', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'(?<!\w)[\^\d+]', '', text)
    text = re.sub(r'[™®©℠]', '', text)
    
    
    fraction_map = {
        '½': '1/2', '⅓': '1/3', '⅔': '2/3', '¼': '1/4', '¾': '3/4',
        '⅕': '1/5', '⅖': '2/5', '⅗': '3/5', '⅘': '4/5', '⅙': '1/6',
        '⅚': '5/6', '⅐': '1/7', '⅛': '1/8', '⅜': '3/8', '⅝': '5/8',
        '⅞': '7/8', '⅑': '1/9', '⅒': '1/10'
    }
    for frac, repl in fraction_map.items():
        text = text.replace(frac, repl)
    
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'So')
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    text = re.sub(r'\s+([,.!?:;])', r'\1', text)
    text = re.sub(r'([,.!?:;])\s+', r'\1 ', text)
    
    text = re.sub(r'\(\s*\)', '', text)  # Removes empty parentheses
    text = re.sub(r'\(\s*[a-z]\s*\)', '', text)  # Removes :(a), :(b), etc.
    text = re.sub(r'\(\s*(Pl\.\s*\d+\s*,)?\s*Fig\.\s*\d+(\.\d+)?\s*\)', '', text)
    text = '\n'.join(line for line in text.split('\n') if len(line.split()) > 1 or len(line.strip()) < 3)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_and_compare(gutenberg_filename, bawe_filename):
    
    gutenberg_df = pd.read_csv(gutenberg_filename)
    bawe_df = pd.read_csv(bawe_filename)
    gutenberg_df['cleaned_text'] = ""
    bawe_df['cleaned_text'] = ""

    print("\nCleaning Gutenberg Data:\n")
    for i in tqdm(range(len(gutenberg_df)), desc="Gutenberg"):
        gutenberg_df.at[i, 'cleaned_text'] = clean_text(gutenberg_df.at[i, 'text'])
        if i % 100 == 0: 
            gutenberg_df[['custom_id', 'author', 'cleaned_text']].to_csv('4493_FromGutenberg_cleaned.csv', index=False)
    
    print("\nCleaning BAWE Data:\n")
    for i in tqdm(range(len(bawe_df)), desc="BAWE"):
        bawe_df.at[i, 'cleaned_text'] = clean_text(bawe_df.at[i, 'text'])
        if i % 100 == 0: 
            bawe_df[['author', 'cleaned_text']].to_csv('BAWE_raw_cleaned.csv', index=False)

    gutenberg_df[['custom_id', 'author', 'cleaned_text']].to_csv('4493_FromGutenberg_cleaned.csv', index=False)
    bawe_df[['author', 'cleaned_text']].to_csv('BAWE_raw_cleaned.csv', index=False)

    print("\nGutenberg Data Cleaning:\n")
    for i, row in gutenberg_df.iterrows():
        original_text = row['text']
        cleaned_text = row['cleaned_text']
        print(f"Original Text:\n{original_text}\n")
        print(f"Cleaned Text:\n{cleaned_text}\n")
        print("-" * 80)
    print("\nBAWE Data Cleaning:\n")
    for i, row in bawe_df.iterrows():
        original_text = row['text']
        cleaned_text = row['cleaned_text']
        print(f"Original Text:\n{original_text}\n")
        print(f"Cleaned Text:\n{cleaned_text}\n")
        print("-" * 80)

gutenberg_file = '4493_FromGutenberg.csv'
bawe_file = 'BAWE_raw.csv'


process_and_compare(gutenberg_file, bawe_file)
