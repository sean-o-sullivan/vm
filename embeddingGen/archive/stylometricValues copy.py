#we should probably try use something like the brown corpus, or that other and better set of authors

#from resource_loader import nlp, english_words
from resource_loader import model, tokenizer, nlp, english_words
import torch
import nltk
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from nltk.tokenize import sent_tokenize
import string
from collections import Counter
#from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import re
#import sklearn
import torch
import numpy as np 
import string
import math
#import os
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import readability2 #this covers the other metrics, like complexity and nominalisation
from readability import Readability #this package is py,readabiliy metrics.
#import json

from nltk.tokenize import sent_tokenize
#from nltk.corpus import wordnet

# Download the words corpus
#nltk.download('wordnet')
#nltk.download('omw-1.4')
nltk.download('punkt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_herdans_v(text, min_val=0, max_val=2.1):
    
    # nltk.download('words')
    # Load the set of English words
    # english_words = set(words.words())
    # nlp = spacy.load('en_core_web_lg')

    doc = nlp(text)

    freq_dist = {}
    for token in doc:
        word = str(token)

        if word in english_words:
            word = word.lower()
            freq_dist[word] = freq_dist.get(word, 0) + 1

    
    V1 = len(freq_dist)
    V2 = sum([count**2 for count in freq_dist.values()])

    
    if V1 != 0:
        V = (math.log(V2) / math.log(V1))
    else:
        V = 0.0

    
    normalized_V = (V - min_val) / (max_val - min_val)
    
    normalized_V = max(0, min(normalized_V, 1))

    return normalized_V

def brunets_w(text, min_val=0, max_val=18.8):
    
    
    words = text.split()

    
    N = len(words)

    
    V = len(set(words))

    
    if V > 0 and N > 0:
        W = N ** (V ** -0.165)
    else:
        W = 0.0

        

    normalized_w = (W - min_val) / (max_val - min_val)
    
    normalized_w = max(0, min(normalized_w, 1))

    return normalized_w

def yules_k_characteristic(text):
    
    def refined_scaling(K, max_K=999, exponent=10, base=15, offset=0):
        
        basic_scaled = math.log(K + offset, base) / \
            math.log(max_K + offset, base)
        return basic_scaled**exponent

    
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    if len(tokens) > 550:
        
        
        for i in range(550, 0, -1):
            if tokens[i] == '.':
                tokens = tokens[:i+1]  
                break
        
        else:
            tokens = tokens[:550]
    text = ' '.join(tokens)  

    
    freq_dist = {}
    for word in tokens:
        word = word.lower()
        freq_dist[word] = freq_dist.get(word, 0) + 1

    
    M1 = sum(freq_dist.values())
    M2 = sum([count**2 for count in freq_dist.values()])

    
    if M1 != 0:
        K = (10**4) * (M1**2 - M2) / M1
    else:
        K = 0.0

    
    max_K = 9999999.0  
    if K > 0:
        scaled_K = refined_scaling(K, max_K)
    else:
        scaled_K = 0.0  

    return scaled_K

def honores_r(text, min_val=0, max_val=10.6):
        
    nltk.download('punkt', quiet=True)

    
    words = nltk.word_tokenize(text)

    
    freq_dist = Counter(words)

    
    N = sum(freq_dist.values())

    
    V = len(freq_dist)

    
    V1 = sum(1 for count in freq_dist.values() if count == 1)

    
    if math.log(N) != 0:
        R = 100 * ((1 - (V1 / V)) / math.log(N))
    else:
        R = 0.0

    
    normalized_R = (R - min_val) / (max_val - min_val)
    
    normalized_R = max(0, min(normalized_R, 1))

    return normalized_R

def renyis_entropy(text, alpha=2, min_val=0, max_val=7):
    
    
    words = word_tokenize(text.lower())

    
    total_words = len(words)
    if total_words == 0:
        return 0  

    
    freq_dist = Counter(words)

    
    probabilities = [count/total_words for count in freq_dist.values()]

    
    if alpha == 1:  
        entropy = -sum(p * math.log(p, 2)
                       for p in probabilities if p > 0)  
    else:
        entropy = (1 / (1 - alpha)) * math.log(sum(p **
                                                   alpha for p in probabilities), 2)  

    
    normalized_entropy = (entropy - min_val) / (max_val - min_val)
    
    normalized_entropy = max(0, min(normalized_entropy, 1))

    return normalized_entropy



def hapax_dislegomena_rate(text, min_val=0, max_val=0.152):
    
    
    
    
    words = text.split()

    
    freq_dist = Counter(words)

    
    total_words = len(words)

    
    if total_words == 0:
        return 0.0

    
    hapax_dislegomena_count = sum(
        1 for count in freq_dist.values() if count == 2)


    
    hapax_dislegomena_rate = hapax_dislegomena_count / total_words


    
    normalized_score = (hapax_dislegomena_rate - min_val) / (max_val - min_val)
    normalized_score = max(0, min(normalized_score, 1))

    return normalized_score


def truncate_to_full_stop(text, tokenizer, max_token_length=1024):
    
    tokenized_text = tokenizer.encode(text)
    
    if len(tokenized_text) <= max_token_length:
        return text
    
    truncated_text = tokenizer.decode(
        tokenized_text[:max_token_length], clean_up_tokenization_spaces=False)
    
    last_full_stop = truncated_text.rfind('.')
    
    if last_full_stop != -1:
        return truncated_text[:last_full_stop + 1]
    
    return truncated_text[:max_token_length]


def perplexity(text, max_length=1024, stride=512, max_value=50):
    
    text = truncate_to_full_stop(text, tokenizer)
    tokenized_text = tokenizer.encode(text)

    
    if len(tokenized_text) > max_length:
        print("Warning: Truncated text is still too long. There may be an error in the truncation function.")
        return None

    
    encodings = tokenizer(text, return_tensors="pt").to(device)

    
    nlls = []

    
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    
    normalized_ppl = min(ppl / max_value, torch.tensor(1.0)).item()

    return normalized_ppl


def average_word_length(text):
    

    max_avg_word_length = 6.9
    
    words = nltk.word_tokenize(text)

    
    words = [word for word in words if word.isalpha()]

    
    total_characters = sum(len(word) for word in words)

    
    total_words = len(words)

    
    average_word_length = total_characters / total_words if total_words > 0 else 0
    normalized_avg_word_length = ""

    
    normalized_avg_word_length = average_word_length / max_avg_word_length

    
    normalized_avg_word_length = min(1, normalized_avg_word_length)


    
    return normalized_avg_word_length

def long_word_rate(text, max_rate=0.45, threshold=8):
    

    
    words = nltk.word_tokenize(text)


    
    words = [word for word in words if word.isalpha()]


    
    total_words = len(words)


    
    long_words = [word for word in words if len(word) >= threshold]


    num_long_words = len(long_words)

    
    long_word_rate = num_long_words / total_words if total_words > 0 else 0


    
    normalized_long_word_rate = long_word_rate / max_rate

    
    normalized_long_word_rate = min(1, normalized_long_word_rate)


    return normalized_long_word_rate

def short_word_rate(text):
    max_rate = 0.58

    threshold = 3
    
    words = nltk.word_tokenize(text)

    
    words = [word for word in words if word.isalpha()]

    
    total_words = len(words)

    
    short_words = [word for word in words if len(word) <= threshold]
    num_short_words = len(short_words)

    
    short_word_rate = num_short_words / total_words if total_words > 0 else 0

    
    normalized_short_word_rate = short_word_rate / max_rate
    
    normalized_short_word_rate = min(1, normalized_short_word_rate)

    return short_word_rate

def lexical_density(text, max_rate=0.69):
    

    
    nltk.download('averaged_perceptron_tagger', quiet=True)


    
    words = nltk.word_tokenize(text)


    
    total_words = len(words)


    
    pos_tags = nltk.pos_tag(words)


    
    lexical_words = [word for word,
                     tag in pos_tags if tag.startswith(('N', 'V', 'J', 'R'))]


    
    num_lexical_words = len(lexical_words)


    
    lexical_density = num_lexical_words / total_words if total_words > 0 else 0


    
    normalized_lex_density = lexical_density / max_rate

    
    normalized_lex_density = min(1, normalized_lex_density)


    return normalized_lex_density

def burstiness(text, min_val=0, max_val=0.95):
    
    
    translator = str.maketrans('', '', string.punctuation.replace('.', ''))
    text = text.lower().translate(translator)

    
    words = re.findall(r'\b\w+\b', text)

    
    if len(words) > 550:
        
        truncated_text = ' '.join(words[:550])
        
        last_full_stop = truncated_text.rfind('.')
        
        if last_full_stop != -1:
            words = words[:last_full_stop]
        else:
            
            words = words[:550]

    
    if len(words) < 2:
        return None

    
    freq_dist = Counter(words)

    
    counts = np.array(list(freq_dist.values()))

    
    mean = np.mean(counts)
    variance = np.var(counts)

    
    if variance == 0:
        return 0

    
    B = (variance - mean) / (variance + mean)

    
    normalized_B = (B - min_val) / (max_val - min_val)
    normalized_B = max(0, min(normalized_B, 1))  

    return normalized_B

def complex_words_rate(text, max_value=0.4):

    
    text = str(text).replace('\n', ' ')

    
    sentences = text.split('. ')

    tokenized_text = '.\n'.join([' '.join(sentence.split())
                                for sentence in sentences])

    
    results = readability2.getmeasures(tokenized_text, lang='en')

    
    complex_words = results['sentence info']['complex_words']
    total_words = results['sentence info']['words']

    
    if total_words > 0:
        rate = complex_words / total_words
        return min(max(rate / max_value, 0), 1)
    else:
        return 0.0

def syll_per_word(raw_text, max_value=2.25):
    def format_text(raw_text):
        paragraphs = raw_text.split('\n\n')
        formatted_text = ''

        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            formatted_paragraph = '\n'.join(sentences) + '\n\n'
            formatted_text += formatted_paragraph

        return formatted_text

    formatted_text = format_text(raw_text)
    results = readability2.getmeasures(formatted_text, lang='en')
    characters_per_word = results['sentence info']['syll_per_word']

    
    normalized_result = min(max(characters_per_word / max_value, 0), 1)

    return normalized_result

def dale_chall_complex_words_rate(text, max_value=0.7):

    
    text = str(text).replace('\n', ' ')

    
    sentences = text.split('. ')

    tokenized_text = '.\n'.join([' '.join(sentence.split())
                                for sentence in sentences])

    
    results = readability2.getmeasures(tokenized_text, lang='en')

    
    dale_chall_complex_words = results['sentence info']['complex_words_dc']
    total_words = results['sentence info']['words']

    
    if total_words > 0:
        rate = dale_chall_complex_words / total_words
        return min(max(rate / max_value, 0), 1)
    else:
        return 0.0





def average_sentence_length(text, upper_limit=50):
        
    nltk.download('punkt', quiet=True)

    
    sentences = nltk.sent_tokenize(text)

    
    words = nltk.word_tokenize(text)

    
    words = [word for word in words if word not in string.punctuation]

    
    total_words = len(words)

    
    total_sentences = len(sentences)

    
    average_sentence_length = total_words / \
        total_sentences if total_sentences > 0 else 0

    
    normalized_average_sentence_length = min(
        average_sentence_length / upper_limit, 1)

    return normalized_average_sentence_length

def frequent_delimiters_rate(text, max_val=0.08105):
    
    
    characters = list(text)

    
    num_commas = characters.count(',')
    num_periods = characters.count('.')

    
    total_characters = len(characters)

    
    
    if total_characters > 0:
        frequent_delimiters_rate = (
            num_commas + num_periods) / total_characters
    else:
        frequent_delimiters_rate = 0.0

    
    normalized_score = frequent_delimiters_rate / max_val
    normalized_score = max(0, min(normalized_score, 1))

    return normalized_score

def lessfrequent_delimiters_rate(text, max_val=0.016):
    
    
    characters = list(text)

    
    num_semicolons = characters.count(';')
    num_colons = characters.count(':')

    
    total_characters = len(characters)

    
    
    if total_characters > 0:
        lessfrequent_delimiters_rate = (
            num_semicolons + num_colons) / total_characters
    else:
        lessfrequent_delimiters_rate = 0.0

    
    normalized_score = lessfrequent_delimiters_rate / max_val
    normalized_score = max(0, min(normalized_score, 1))

    return normalized_score

def parentheticals_and_brackets_rate(text, max_rate=0.037):
    
    
    characters = list(text)

    
    num_open_parentheses = characters.count('(')
    num_close_parentheses = characters.count(')')
    num_open_brackets = characters.count('[')
    num_close_brackets = characters.count(']')

    
    total_characters = len(characters)

    
    
    if total_characters > 0:
        parentheticals_rate = (num_open_parentheses + num_close_parentheses +
                               num_open_brackets + num_close_brackets) / total_characters
    else:
        parentheticals_rate = 0.0

    
    normalized_rate = min(parentheticals_rate / max_rate, 1.0)

    return normalized_rate

def quotations_rate(text, max_rate=0.0214):
    
    
    characters = list(text)

    
    num_single_quotes = characters.count("'")
    num_double_quotes = characters.count('"')

    
    total_characters = len(characters)

    
    
    if total_characters > 0:
        quotations_rate = (num_single_quotes +
                           num_double_quotes) / total_characters
    else:
        quotations_rate = 0.0

        
    normalized_rate = min(quotations_rate / max_rate, 1.0)

    return normalized_rate

def dashes_and_ellipses_rate(text, max_rate=0.028):
    
    
    characters = list(text)

    
    num_hyphens = characters.count('-')
    num_en_dashes = characters.count('–')
    num_em_dashes = characters.count('—')
    num_ellipses = text.count('...')

    
    total_characters = len(characters)

    
    
    if total_characters > 0:
        dashes_and_ellipses_rate = (
            num_hyphens + num_en_dashes + num_em_dashes + num_ellipses) / total_characters
    else:
        dashes_and_ellipses_rate = 0.0

        
    normalized_rate = min(dashes_and_ellipses_rate / max_rate, 1.0)

    return normalized_rate





def flesch_reading_ease(text, min_val=-20, max_val=86.6):
    
    words = text.split()

    
    while len(words) < 100:
        words.extend(words)
        text = " ".join(words)

    r = Readability(text)
    F = r.flesch().score

    
    normalized_score = (F - min_val) / (max_val - min_val)
    normalized_score = max(0, min(normalized_score, 1))

    return normalized_score

def GFI(text, min_val=4, max_val=30):

    words = text.split()

    
    while len(words) < 100:
        words.extend(words)
        text = " ".join(words)

    r = Readability(text)
    GFI = r.gunning_fog().score

    
    
    normalized_score = (GFI - min_val) / (max_val - min_val)
    normalized_score = max(0, min(normalized_score, 1))

    return normalized_score

def compute_coleman_liau_score(text):
    words = text.split()

    
    while len(words) < 100:
        words.extend(words)
        text = " ".join(words)

    r = Readability(text)
    return r.coleman_liau().score


def coleman_liau_index(text, min_val=0, max_val=21):
    
    
    CLI = compute_coleman_liau_score(text)

    
    normalized_score = (CLI - min_val) / (max_val - min_val)
    normalized_score = max(0, min(normalized_score, 1))

    return normalized_score


def compute_ari_score(text):
    words = text.split()

    
    while len(words) < 100:
        words.extend(words)
        text = " ".join(words)

    r = Readability(text)
    return r.ari().score


def ari(text, min_val=0, max_val=37.35):
        
    ARI = compute_ari_score(text)

    
    normalized_score = (ARI - min_val) / (max_val - min_val)
    normalized_score = max(0, min(normalized_score, 1))

    return normalized_score



def dale_chall(text, min_val=0, max_val=18.54):

    words = text.split()

    
    while len(words) < 100:
        words.extend(words)
        text = " ".join(words)

    r = Readability(text)
    DC = r.dale_chall().score

    
    normalized_score = (DC - min_val) / (max_val - min_val)
    normalized_score = max(0, min(normalized_score, 1))

    return normalized_score

def lix(text, min_val=0, max_val=80):

    words = text.split()

    
    while len(words) < 100:
        words.extend(words)
        text = " ".join(words)

    
    
    sentences = [s.strip() for s in text.split('.') if s]
    words = [word for word in text.split() if word not in string.punctuation]

    
    long_words = [word for word in words if len(word) > 6]

    
    if len(sentences) > 0 and len(words) > 0:

        lix_score = (len(words) / len(sentences)) + \
            (len(long_words) / len(words) * 100)

    else:
        lix_score = 0.0

    
    normalized_score = (lix_score - min_val) / (max_val - min_val)
    normalized_score = max(0, min(normalized_score, 1))

    return normalized_score






def nominalization(text, max_value=0.151):

    
    text = str(text).replace('\n', ' ')

    
    sentences = text.split('. ')
    tokenized_text = '.\n'.join([' '.join(sentence.split())
                                for sentence in sentences])

    
    results = readability2.getmeasures(tokenized_text, lang='en')

    
    nomi = results['word usage']['nominalization']
    words = results['sentence info']['words']

    nominalization = nomi/words

    return min(max(nominalization / max_value, 0), 1)

def preposition_usage(text):  

    max_values = {
        'in': 0.07200000000000005,
        'of': 0.10700000000000008,
        'to': 0.07700000000000004,
        'for': 0.04500000000000003,
        'with': 0.04000000000000003,
        'on': 0.04300000000000003,
        'at': 0.05300000000000004,
        'by': 0.03200000000000002,
        'from': 0.048000000000000036,
        'up': 0.03200000000000002,
        'about': 0.016000000000000007,
        'into': 0.02000000000000001,
        'over': 0.02000000000000001,
        'after': 0.04000000000000003}

    def preprocess_text(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.lower().translate(translator).split()

    
    prepositions = ['in', 'of', 'to', 'for', 'with', 'on', 'at',
                    'by', 'from', 'up', 'about', 'into', 'over', 'after']

    
    words = preprocess_text(text)

    
    word_count = Counter(words)
    total_words = sum(word_count.values())

    
    normalized_rates_list = []

    
    for preposition in prepositions:
        raw_rate = word_count[preposition] / \
            total_words if total_words > 0 else 0
        
        
        max_value = max_values.get(preposition, 0.06)
        normalized_rate = min(max(raw_rate / max_value, 0), 1)
        normalized_rates_list.append(normalized_rate)

    return normalized_rates_list

def conjunctions_usage(text):
    
    max_values = {'and': 0.08400000000000006,
                  'or': 0.02900000000000002,
                  'but': 0.035000000000000024,
                  'so': 0.03100000000000002,
                  'for': 0.04400000000000003}
    

    def preprocess_text(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.lower().translate(translator).split()

    
    def calculate_conjunction_usage_rate(words, conjunction, max_value):
        word_count = Counter(words)
        total_words = sum(word_count.values())
        conjunction_count = word_count[conjunction]
        usage_rate = conjunction_count / total_words if total_words > 0 else 0
        
        clamped_rate = min(usage_rate, max_value) / max_value
        return clamped_rate

    
    words = preprocess_text(text)

    
    conjunctions = ['and', 'or', 'but', 'so', 'for']

    
    usage_rates = [calculate_conjunction_usage_rate(words, conjunction, max_values.get(
        conjunction, 0.06)) for conjunction in conjunctions]

    return usage_rates

def auxiliary_infipart_modals_usage_rate(text):

    
    max_values = {
        'group1_auxiliaries': 0.08500000000000006,
        'group2_auxiliaries': 0.04100000000000003,
        'infinitives_participles': 0.053000000000000026,
        'can': 0.061000000000000008,
        'could': 0.02100000000000001,
        'should': 0.0218000000000000008,
        'will': 0.04200000000000003,
        'would': 0.02900000000000002,
        'may': 0.032000000000000003,
        'might': 0.012000000000000004,
        'must': 0.034000000001}

    
    def preprocess_text(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.lower().translate(translator).split()

    
    def calculate_function_word_usage_rate(words, function_words, max_value):
        word_count = Counter(words)
        total_words = sum(word_count.values())
        function_word_count = sum(word_count[word] for word in function_words)
        raw_rate = function_word_count / total_words if total_words > 0 else 0
        normalized_rate = min(raw_rate / max_value, 1)
        return normalized_rate

    
    group1_auxiliaries = {'is', 'are', 'was', 'were', 'do', 'does', 'did'}
    group2_auxiliaries = {'have', 'has', 'had'}

    
    infinitives_participles = {'be', 'being', 'been'}

    
    modals = ['can', 'could', 'should', 'will',
              'would', 'may', 'might', 'must']

    
    words = preprocess_text(text)

    
    usage_rates = [
        calculate_function_word_usage_rate(
            words, group1_auxiliaries, max_values['group1_auxiliaries']),
        calculate_function_word_usage_rate(
            words, group2_auxiliaries, max_values['group2_auxiliaries']),
        calculate_function_word_usage_rate(
            words, infinitives_participles, max_values['infinitives_participles']),
    ]
    usage_rates.extend(calculate_function_word_usage_rate(
        words, {modal}, max_values[modal]) for modal in modals)
    return usage_rates






