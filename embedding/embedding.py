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
from readability import Readability
#import json

import readability2



from readability import Readability

from nltk.tokenize import sent_tokenize
#from nltk.corpus import wordnet

# Download the words corpus
#nltk.download('wordnet')
#nltk.download('omw-1.4')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
nltk.data.path.append('/usr/local/nltk_data')

from stylometricValues import *


def Embedding(text):
    """
    Generates an embedding of the text based on the selected categories.

    Parameters:
    text (str): The input text.

    Returns:
    list: The embedding of the text.
    """
    embedding = []
    doc = process_text(text)

    embedding.extend([
        compute_herdans_v(text),  # I Feature
        brunets_w(text),  # I Feature
        yules_k_characteristic(text),  # I Feature
        honores_r(text),  # I Feature
        renyis_entropy(text),  # I Feature
        hapax_dislegomena_rate(text),  # I Feature
        perplexity(text),  # I Feature
        burstiness(text),  # I Feature
        average_word_length(text),  # I Feature
        long_word_rate(text),  # I Feature
        short_word_rate(text),  # I Feature
        lexical_density(text),  # I Feature
        complex_words_rate(text),
        dale_chall_complex_words_rate(text),
        syll_per_word(text),
        average_sentence_length(text),  # I Feature
        frequent_delimiters_rate(text),  # I Feature
        lessfrequent_delimiters_rate(text),  # I Feature

        parentheticals_and_brackets_rate(text),  # I Feature
        quotations_rate(text),  # I Feature
        dashes_and_ellipses_rate(text),  # I Feature
        flesch_reading_ease(text),  # I Feature
        GFI(text),  # I Feature
        coleman_liau_index(text),  # I Feature
        ari(text),  # I Feature
        dale_chall(text),  # I Feature
        lix(text),  # I Feature
    ])

    embedding.extend([nominalization(text)])

        # XIV Features       prepositions = ['in', 'of', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'up', 'about', 'into', 'over', 'after']
    embedding.extend(preposition_usage(text))

        # V Features      conjunctions = ['and', 'or', 'but', 'so', 'for']
    embedding.extend(conjunctions_usage(text))

    embedding.extend(auxiliary_infipart_modals_usage_rate(text))
        # This functon includes the following values: (1) group1_auxiliaries = {'is', 'are', 'was', 'were', 'do', 'does', 'did'}
        # (2) group2_auxiliaries = {'have', 'has', 'had'}
        # (3) infinitives_participles = {'be', 'being', 'been'}
        # (4-8)  modals = ['can', 'could', 'should', 'will', 'would', 'may', 'might', 'must']

    return embedding

    # V Features - Detailed Conjunctions Usage
    conjunctions_rates = detailed_conjunctions_usage(doc)
    embedding.extend([
        conjunctions_rates["coordinating"],
        conjunctions_rates["subordinating"],
        conjunctions_rates["correlative"]
    ])


