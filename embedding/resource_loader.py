from flask import Flask
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import spacy
import nltk
from nltk.corpus import words


nlp = spacy.load('en_core_web_lg')

english_words = set(words.words())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'gpt2-large'

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')





