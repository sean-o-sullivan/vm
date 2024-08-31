import re
import unicodedata
from bs4 import BeautifulSoup
from collections import Counter

def find_true_end(text, initial_end_pos, lookahead_range=1000):
    current_end_pos = initial_end_pos
    while True:
        lookahead_text = text[current_end_pos:current_end_pos + lookahead_range]
        next_end_match = re.search(r'([=]{5,}|[-]{5,}|[.]{5,}|-{10,})', lookahead_text)
        if next_end_match:
            current_end_pos += next_end_match.end()
        else:
            break
    return current_end_pos

def remove_table_from_text(text, stats):
    cleaned_text = ""
    position = 0
    removed_tables = []
    while True:
        start_match = re.search(r'\b[A-Z]{5,}\b', text[position:])
        if not start_match:
            cleaned_text += text[position:]
            break
        start_pos = position + start_match.start()
        cleaned_text += text[position:start_pos].strip() + "\n"
        lookahead_range = 500
        lookahead_text = text[start_pos:start_pos + lookahead_range]
        table_end_match = re.search(r'([=]{5,}|[-]{5,}|[.]{5,}|-{10,})', lookahead_text)
        if not table_end_match:
            position = start_pos + len(start_match.group(0))
            continue
        initial_end_pos = start_pos + table_end_match.end()
        true_end_pos = find_true_end(text, initial_end_pos)
        table_content = text[start_pos:true_end_pos]
        removed_tables.append(table_content)
        stats['tables_removed'] += 1
        position = true_end_pos
    return cleaned_text.strip(), removed_tables

def clean_text(text, use_table=False):
    stats = Counter()
    removed_tables = []
    soup = BeautifulSoup(text, 'html.parser')
    stats['html_tags_removed'] = len(list(soup.find_all()))
    text = soup.get_text()
    text, n = re.subn(r'\((?:\w+,?\s+(?:et al\.)?,?\s+)?(?:19|20)\d{2}[a-z]?(?::\d+(?:-\d+)?)?(?:\s+and\s+(?:\w+,?\s+(?:et al\.)?,?\s+)?(?:19|20)\d{2}[a-z]?(?::\d+(?:-\d+)?)?)*\)', '', text)
    stats['citations_removed'] += n
    text, n = re.subn(r'\[.*?\]', '', text)
    stats['square_brackets_removed'] += n

    if use_table:
        text, tables = remove_table_from_text(text, stats)
        removed_tables.extend(tables)

    text, n = re.subn(r'\{.*?\}', '', text)
    stats['curly_braces_removed'] += n
    patterns = {
        r'\*+': 'asterisks_removed',
        r'(?m)^\s*[\|+].*[\|+]\s*$': 'table_like_structures_removed',
        r'(?m)^\s*[-+]+\s*$': 'table_like_structures_removed',
        r'(?m)^\s*[a-zA-Z0-9]+\s*[-+*/^()]+.*$': 'equations_removed',
        r'[±∓×÷∙∘·°∂∇∆∑∏∫√∛∜∝∞≈≠≡≤≥≪≫⊂⊃⊄⊅⊆⊇⊈⊉⊊⊋∈∉∋∌∍∎∏∐∑−]': 'special_characters_removed'
    }

    for pattern, stat_key in patterns.items():
        text, n = re.subn(pattern, '', text)
        stats[stat_key] += n

    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    words_to_remove = ["Introduction", "Summary", "Abstract", "Objective", "Executive Summary"]
    for word in words_to_remove:
        text, n = re.subn(r'\b' + re.escape(word) + r'\b', '', text)
        stats[f'{word}_removed'] = n

    text, n = re.subn(r'([!?.]){2,}', r'\1', text)
    stats['repeated_punctuation_removed'] += n
    text, n = re.subn(r'\s+([,.!?:;])', r'\1', text)
    text, n = re.subn(r'([,.!?:;])\s+', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text, stats, removed_tables

sample_text_1 = "Your first sample text goes here."
cleaned_text_1, stats_1, removed_tables_1 = clean_text(sample_text_1, use_table=True)
print(f"Cleaned Text 1:\n{cleaned_text_1}\n")
print(f"Stats 1:\n{stats_1}\n")
if removed_tables_1:
    print(f"Removed Tables 1:\n{removed_tables_1}\n")
