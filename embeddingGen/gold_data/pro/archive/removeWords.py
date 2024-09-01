import pandas as pd
import re
from tqdm import tqdm

def remove_words(text, words_to_remove):
    for word in words_to_remove:
        pattern = r'(?<!\S)' + re.escape(word) + r'(?!\S)'
        text = re.sub(pattern, '', text)
    return text
words_to_remove = [
    "'Introduction", "'Summary", "'Abstract", "'Objective", "'Executive Summary", 
    "'Aim:", "'Referral informationR", "'PART: EVALUATION", "' Introduction", 
    "'SITUATION", "'AIM", "'INTRODUCTION", "'Project PlanningI", 
    "') Selfish Genes and Group Selection", "'Part - A", "'a)", " b) ", " c) ", 
    " d) ", " e) ", "(see figure ) ", "(Figure ) ", "( ", "FORMULA"
]
input_file = 'BAWE_Clean.csv'
output_file = 'BAWE_words_removed.csv'
df = pd.read_csv(input_file)
print("Removing words from BAWE corpus:")
tqdm.pandas()
df['cleaned_text'] = df['cleaned_text'].progress_apply(lambda x: remove_words(x, words_to_remove))

df.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")
