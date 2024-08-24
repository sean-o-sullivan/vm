import stanza  
import re  
import math  
import numpy as np  
import textstat
from readability import getmeasures
from collections import Counter, defaultdict

stanza.download('en')  

nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse,ner')
from wordfreq import word_frequency
from scipy.optimize import curve_fit  
from wordtangible import word_concreteness, avg_text_concreteness, concrete_abstract_ratio
import nltk
nltk.download('punkt_tab')
import pronouncing

def process_text(text):
    return nlp(text)

def compute_herdans_v(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    unique_words = len(set(words))
    
    if total_words == 0 or unique_words == 0:
        return 0
    
    
    herdans_v = math.log(unique_words) / math.log(total_words)
    
    return herdans_v

def compute_brunets_w(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    
    unique_words = len(set(words))
    
    
    if total_words == 0 or unique_words == 0:
        return 0.0
    
    
    a = 0.165
    
    
    brunets_w = total_words ** (unique_words ** (-a))
    
    return brunets_w

def tuldava_ln_ttr(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    
    unique_words = len(set(words))
    
    if total_words == 0 or unique_words == 0:
        return 0.0
    
    
    ln_ttr = math.log(unique_words) / math.log(total_words)
    
    return ln_ttr

def simpsons_index(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 1.0  

    
    word_freqs = Counter(words)
    
    
    simpsons_d = sum((count / total_words) ** 2 for count in word_freqs.values())
    
    
    return 1 - simpsons_d

def sichel_s_measure(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0  

    
    unique_words = len(set(words))
    
    if unique_words == 0:
        return 0  
    
    
    sichel_s = (unique_words - math.log10(unique_words)) / math.sqrt(total_words)
    
    return sichel_s

def orlov_sigma(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0  

    
    word_freqs = Counter(words)
    
    
    average_freq = total_words / len(word_freqs)
    
    
    sum_squared_deviations = sum((freq - average_freq) ** 2 for freq in word_freqs.values())
    
    
    orlov_sigma = math.sqrt(sum_squared_deviations / len(word_freqs))
    
    return orlov_sigma

def pos_frequencies(doc):
        
    pos_tags = [word.upos for sentence in doc.sentences for word in sentence.words]
    
    
    pos_freqs = Counter(pos_tags)
    
    
    total_pos_tags = len(pos_tags)
    pos_frequencies = {tag: freq / total_pos_tags for tag, freq in pos_freqs.items()}
    
    return pos_frequencies

def clauses_per_sentence(doc):

    print(type(doc))

    total_clauses = 0
    total_sentences = len(doc.sentences)
    
    if total_sentences == 0:
        return 0  

    
    for sentence in doc.sentences:
        
        
        clause_relations = {'acl', 'ccomp', 'xcomp', 'advcl', 'relcl', 'nmod'}
        num_clauses = sum(word.deprel in clause_relations for word in sentence.words)
        
        total_clauses += num_clauses
    
    
    avg_clauses_per_sentence = total_clauses / total_sentences
    
    return avg_clauses_per_sentence

def modifiers_per_noun_phrase(doc):
    total_modifiers = 0
    total_noun_phrases = 0
    
    
    modifier_relations = {'amod', 'nmod', 'acl', 'advmod', 'det', 'appos'}
    
    
    for sentence in doc.sentences:
        noun_phrases = [word for word in sentence.words if word.upos == 'NOUN' or word.upos == 'PROPN']
        
        if not noun_phrases:
            continue
        
        total_noun_phrases += len(noun_phrases)
        
        
        for noun_phrase in noun_phrases:
            modifiers = [word for word in sentence.words if word.deprel in modifier_relations and word.head == noun_phrase.id]
            total_modifiers += len(modifiers)
    
    
    if total_noun_phrases == 0:
        return 0  

    avg_modifiers_per_noun_phrase = total_modifiers / total_noun_phrases
    
    return avg_modifiers_per_noun_phrase

def coordinated_phrases_per_sentence(doc):
    total_coordinated_phrases = 0
    total_sentences = len(doc.sentences)
    
    if total_sentences == 0:
        return 0  

    
    for sentence in doc.sentences:
        
        words = sentence.words
        
        
        coordinating_conjunctions = {'cc'}
        
        
        coordinated_phrases = 0
        for word in words:
            if word.deprel in coordinating_conjunctions:
                
                
                coordinated_phrases += 1
        
        total_coordinated_phrases += coordinated_phrases
    
    
    avg_coordinated_phrases_per_sentence = total_coordinated_phrases / total_sentences
    
    return avg_coordinated_phrases_per_sentence

def sentence_length_variation(doc):
    sentence_lengths = [len(sentence.words) for sentence in doc.sentences]
    
    if not sentence_lengths:
        return 0  
    
    std_dev = np.std(sentence_lengths)

    return std_dev

def maas_index(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    
    unique_words = len(set(words))
    
    if total_words == 0 or unique_words == 0:
        return 0.0
    
    
    maas_index_value = (math.log(total_words) - math.log(unique_words)) / (math.log(total_words) ** 2)
    
    return maas_index_value

def extract_clauses(doc):
 
    clauses = []
    for sentence in doc.sentences:
        current_clause = []
        for word in sentence.words:
            if word.deprel in {'ccomp', 'xcomp', 'acl', 'advcl', 'relcl'}:
                if current_clause:
                    clauses.append(current_clause)
                current_clause = []
            current_clause.append(word)
        if current_clause:
            clauses.append(current_clause)
    return clauses

def clause_length_variation(doc):

    clauses = extract_clauses(doc)
    clause_lengths = [len(clause) for clause in clauses]
    
    if not clause_lengths:
        return 0  
    
    
    std_dev = np.std(clause_lengths)
    return std_dev

def subordination_index(doc):

    clauses = extract_clauses(doc)
    total_clauses = len(clauses)
    subordinate_clauses = sum(1 for clause in clauses if any(word.deprel in {'ccomp', 'xcomp', 'acl', 'advcl', 'relcl'} for word in clause))
    
    if total_clauses == 0:
        return 0
    
    
    subordination_index = subordinate_clauses / total_clauses
    return subordination_index

def compute_sentence_depth(sentence):
 
    def get_depth(word_id, depth_map):
        if word_id in depth_map:
            return depth_map[word_id]
        
        word = sentence.words[word_id - 1]  
        if word.head == 0:  
            depth = 0
        else:
            depth = 1 + get_depth(word.head, depth_map)
        
        depth_map[word_id] = depth
        return depth

    depth_map = {}
    max_depth = 0
    for word in sentence.words:
        depth = get_depth(word.id, depth_map)
        max_depth = max(max_depth, depth)

    return max_depth

def average_sentence_depth(doc):
    depths = []
    
    for sentence in doc.sentences:
        depth = compute_sentence_depth(sentence)
        depths.append(depth)
    
    
    if len(depths) == 0:
        return 0
    
    average_depth = sum(depths) / len(depths)
    
    return average_depth

def extract_coordinate_phrases(doc):
    coordinate_phrases = []
    total_phrases = []
    coord_conjunctions = {'and', 'or', 'but', 'nor', 'for', 'so', 'yet'}
    
    current_phrase = []
    
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text.lower() in coord_conjunctions:
                if current_phrase:
                    total_phrases.append(' '.join(current_phrase))
                    current_phrase = []
                coordinate_phrases.append(word.text)
            else:
                current_phrase.append(word.text)
        
        if current_phrase:
            total_phrases.append(' '.join(current_phrase))
            current_phrase = []
    
    return coordinate_phrases, total_phrases

def coordinate_phrases_ratio(doc):
    coordinate_phrases, total_phrases = extract_coordinate_phrases(doc)

    num_coordinate_phrases = len(coordinate_phrases)
    num_total_phrases = len(total_phrases)
    
    if num_total_phrases == 0:
        return 0
    
    
    ratio = num_coordinate_phrases / num_total_phrases
    
    return ratio

def extract_dependent_clauses(doc):
    dependent_clauses = []
    dep_clause_rels = {'ccomp', 'xcomp', 'acl', 'advcl', 'relcl', 'mark'}

    current_clause = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.deprel in dep_clause_rels:
                if current_clause:
                    dependent_clauses.append(' '.join(current_clause))
                    current_clause = []
                dependent_clauses.append(word.text)
            else:
                current_clause.append(word.text)
        
        if current_clause:
            dependent_clauses.append(' '.join(current_clause))
            current_clause = []

    return dependent_clauses

def dependent_clauses_ratio(doc):
        
    
    dependent_clauses = extract_dependent_clauses(doc)
    
    
    total_clauses = len(dependent_clauses)  
    
    
    all_clauses = [sentence.text for sentence in doc.sentences]
    num_total_clauses = len(all_clauses)
    
    if num_total_clauses == 0:
        return 0
    
    
    ratio = len(dependent_clauses) / num_total_clauses
    
    return ratio

def dugast_vmi(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    
    unique_words = len(set(words))
    
    if total_words == 0:
        return 0.0
    
    
    vmi = (unique_words ** 2) / total_words
    
    return vmi

def yules_k_characteristic(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0
    
    
    frequency_distribution = {}
    for word in words:
        if word in frequency_distribution:
            frequency_distribution[word] += 1
        else:
            frequency_distribution[word] = 1
    
    
    M1 = total_words
    
    
    M2 = sum(freq ** 2 for freq in frequency_distribution.values())
    
    
    yules_k = 10**4 * (M1 - M2) / (M2 ** 2)
    
    return yules_k

def honores_r(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    
    unique_words = len(set(words))
    
    
    hapax_legomena = sum(1 for word in set(words) if words.count(word) == 1)
    
    
    if total_words == 0 or unique_words == 0:
        return 0.0
    
    
    honores_r_value = 100 * (math.log(total_words)) / (1 - (hapax_legomena / unique_words))
    
    return honores_r_value

def renyis_entropy(doc, alpha=2):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    
    frequency_distribution = {}
    for word in words:
        if word in frequency_distribution:
            frequency_distribution[word] += 1
        else:
            frequency_distribution[word] = 1
    
    
    probabilities = [freq / total_words for freq in frequency_distribution.values()]
    
    
    if alpha == 1:
        
        entropy = -sum(p * math.log2(p) for p in probabilities)
    else:
        entropy_sum = sum(p ** alpha for p in probabilities)
        entropy = (1 / (1 - alpha)) * math.log2(entropy_sum)
    
    return entropy

def hapax_dislegomena_rate(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    
    frequency_distribution = {}
    for word in words:
        if word in frequency_distribution:
            frequency_distribution[word] += 1
        else:
            frequency_distribution[word] = 1
    
    
    hapax_dislegomena = sum(1 for freq in frequency_distribution.values() if freq == 2)
    
    
    hapax_dislegomena_rate = hapax_dislegomena / total_words
    
    return hapax_dislegomena_rate

def perplexity(doc):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return float('inf')  
    
    
    frequency_distribution = {}
    for word in words:
        if word in frequency_distribution:
            frequency_distribution[word] += 1
        else:
            frequency_distribution[word] = 1
    
    
    probabilities = [freq / total_words for freq in frequency_distribution.values()]
    
    
    perplexity = 2 ** (-sum(p * math.log2(p) for p in probabilities) / total_words)
    
    return perplexity

def burstiness(doc, num_segments=10):
        
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    
    segment_length = total_words // num_segments
    if segment_length == 0:
        return 0.0  
    
    
    segments = [words[i:i + segment_length] for i in range(0, total_words, segment_length)]
    
    
    burstiness_values = []
    for word in set(words):
        counts = [segment.count(word) for segment in segments]
        if np.mean(counts) > 0:  
            burstiness_values.append(np.var(counts) / np.mean(counts))
    
    
    average_burstiness = np.mean(burstiness_values) if burstiness_values else 0.0
    
    return average_burstiness

def average_word_length(doc):
        
    words = [word.text for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    
    total_length = sum(len(word) for word in words)
    
    
    average_length = total_length / total_words
    
    return average_length

def long_word_rate(doc, length_threshold=7):
        
    words = [word.text for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    
    long_words_count = sum(1 for word in words if len(word) >= length_threshold)
    
    
    long_word_rate = long_words_count / total_words
    
    return long_word_rate

def short_word_rate(doc, length_threshold=4):
        
    words = [word.text for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    
    short_words_count = sum(1 for word in words if len(word) < length_threshold)
    
    
    short_word_rate = short_words_count / total_words
    
    return short_word_rate

def lexical_density(doc):
        
    content_word_tags = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    
    total_words = 0
    content_words = 0
    
    for sentence in doc.sentences:
        for word in sentence.words:
            total_words += 1
            if word.upos in content_word_tags:
                content_words += 1
    
    if total_words == 0:
        return 0.0
    
    
    lexical_density = content_words / total_words
    
    return lexical_density

def is_passive_voice(sentence):

    
    has_auxpass = any(word.deprel == 'auxpass' for word in sentence.words)
    has_vbn = any(word.upos == 'VERB' and word.feats and 'VerbForm=Part' in word.feats for word in sentence.words)

    return has_auxpass and has_vbn

def ratio_of_passive_voice(doc):
    sentences = doc.sentences
    
    if not sentences:
        return 0.0
    
    
    passive_count = sum(1 for sentence in sentences if is_passive_voice(sentence))
    
    
    ratio = passive_count / len(sentences)
    
    return ratio

def is_cleft_sentence(sentence):
        
    words = [word.text.lower() for word in sentence.words]
    pos_tags = [word.pos for word in sentence.words]

    
    cleft_indicators = ['it', 'that', 'was', 'is']
    
    
    if 'it' in words and 'that' in words:
        it_index = words.index('it')
        that_index = words.index('that')
        
        
        if it_index < that_index and any(word in pos_tags for word in ['AUX', 'VERB']):
            return True
    
    return False

def ratio_of_cleft_sentences(doc):
    sentences = doc.sentences
    
    if not sentences:
        return 0.0
    
    cleft_count = sum(1 for sentence in sentences if is_cleft_sentence(sentence))
    
    ratio = cleft_count / len(sentences)
    
    return ratio

def count_assonance(text):

    vowels = 'aeiou'
    assonance_count = 0
    words = text.split()
    
    
    phonetic_words = {word: pronouncing.phones_for_word(word) for word in words}
    
    for i in range(len(words) - 1):
        word1, word2 = words[i].lower(), words[i + 1].lower()
        
        
        phones1 = phonetic_words.get(word1, [])
        phones2 = phonetic_words.get(word2, [])
        
        
        for phone1 in phones1:
            for phone2 in phones2:
                if any(vowel in phone1 and vowel in phone2 for vowel in vowels):
                    assonance_count += 1
                    break
    
    return assonance_count

def normalized_assonance(text):

    assonance_count = count_assonance(text)
    num_words = len(text.split())
    
    return assonance_count / num_words if num_words > 0 else 0.0

def count_alliteration(text):
 
    alliteration_count = 0
    words = text.split()
    
    
    phonetic_words = {word: pronouncing.phones_for_word(word) for word in words}
    
    for i in range(len(words) - 1):
        initial1 = get_initial_consonant(words[i])
        initial2 = get_initial_consonant(words[i + 1])
        
        if initial1 and initial2 and initial1 == initial2:
            alliteration_count += 1
    
    return alliteration_count

def get_initial_consonant(word):

    phones = pronouncing.phones_for_word(word)
    if phones:
        phonetic_rep = phones[0]
        match = re.match(r'[^aeiou\W]*', phonetic_rep)
        return match.group(0) if match else ''
    return ''

def normalized_alliteration(text):

    alliteration_count = count_alliteration(text)
    num_words = len(text.split())
    
    return alliteration_count / num_words if num_words > 0 else 0.0

def tempo_variation(doc):

    syllable_counts_per_sentence = []

    for sentence in doc.sentences:
        syllable_count = sum(len(pronouncing.phones_for_word(word.text)) for word in sentence.words if pronouncing.phones_for_word(word.text))
        syllable_counts_per_sentence.append(syllable_count)

    if len(syllable_counts_per_sentence) > 1:
        return np.std(syllable_counts_per_sentence)
    return 0.0

def rhythmic_complexity(text):

    words = text.split()
    stress_patterns = []

    def get_stress_pattern(word):
        phones = pronouncing.phones_for_word(word)
        if phones:
            stress_pattern = pronouncing.stresses(phones[0])
            return [int(s == '1') for s in stress_pattern]
        return []

    for word in words:
        stress_patterns.extend(get_stress_pattern(word))

    if len(stress_patterns) > 1:
        return np.std(stress_patterns)
    return 0.0

def prosodic_patterns(doc):

    sentence_lengths = []

    for sentence in doc.sentences:
        sentence_length = len(sentence.words)
        sentence_lengths.append(sentence_length)

    if len(sentence_lengths) > 1:
        return np.std(sentence_lengths)
    return 0.0

def is_fronted_adverbial(word, sentence, pos_tags=None, dep_rels=None, clause_boundary_tags=None):

    
    if pos_tags is None:
        pos_tags = {"VERB", "AUX"}
    if dep_rels is None:
        dep_rels = {"advmod", "nmod", "obl", "prep", "mark", "acl", "advcl", "xcomp"}
    if clause_boundary_tags is None:
        clause_boundary_tags = {"CCONJ", "PUNCT"}  

    
    if word.id == 1 or word.id == 2:
        
        if hasattr(word, 'deprel') and hasattr(word, 'head'):
            head_word = sentence[word.head - 1] if isinstance(word.head, int) and 0 < word.head <= len(sentence) else None
            if word.deprel in dep_rels and head_word and hasattr(head_word, 'upos') and head_word.upos in pos_tags:
                return True

    
    for i, w in enumerate(sentence[:5]):  
        if hasattr(w, 'deprel') and hasattr(w, 'head'):
            head_word = sentence[w.head - 1] if isinstance(w.head, int) and 0 < w.head <= len(sentence) else None
            if w.deprel in dep_rels and head_word and hasattr(head_word, 'upos') and head_word.upos in pos_tags:
                if i == 0 or (i > 0 and hasattr(sentence[i-1], 'upos') and sentence[i-1].upos in clause_boundary_tags):
                    return True

    return False

def calculate_fronted_adverbial_ratio(doc, pos_tags=None, dep_rels=None, clause_boundary_tags=None):
    """
    Calculate the ratio of sentences with fronted adverbials in a given text,
    with flexibility for POS tags, dependency relations, and clause boundary analysis.
    """
    fronted_adverbials = 0
    total_sentences = len(doc.sentences)

    for sent in doc.sentences:
        
        if is_fronted_adverbial(sent.words[0], sent.words, pos_tags, dep_rels, clause_boundary_tags):
            fronted_adverbials += 1
        else:
            
            for word in sent.words[1:5]:  
                if is_fronted_adverbial(word, sent.words, pos_tags, dep_rels, clause_boundary_tags):
                    fronted_adverbials += 1
                    break

    
    ratio = fronted_adverbials / total_sentences if total_sentences > 0 else 0
    return ratio
def is_inverted_structure(sentence):

    
    subject_position = None
    verb_position = None

    for i, word in enumerate(sentence.words):
        if word.deprel in ('nsubj', 'nsubjpass'):
            subject_position = i
        elif word.deprel == 'root':
            verb_position = i

    
    if subject_position is not None and verb_position is not None:
        
        if subject_position > verb_position:
            return True

    return False

def ratio_of_inverted_structures(doc):
    sentences = doc.sentences
    
    if not sentences:
        return 0.0
    
    inverted_count = sum(1 for sentence in sentences if is_inverted_structure(sentence))
    
    ratio = inverted_count / len(sentences)
    
    return ratio

def is_initial_conjunction(token, sentence):
    
    if token.id == sentence[0].id:
        
        if token.pos == 'CC':
            return True
    return False

def calculate_initial_conjunction_ratio(doc):

    initial_conjunctions = 0
    total_sentences = len(doc.sentences)

    for sent in doc.sentences:
        for token in sent.tokens:
            if is_initial_conjunction(token, sent.tokens):
                initial_conjunctions += 1
                break  

    
    ratio = initial_conjunctions / total_sentences if total_sentences > 0 else 0
    return ratio

def is_embedded_clause(token):
    return token.deprel in {"acl", "relcl", "ccomp", "xcomp", "nsubj"}

def calculate_embedded_clause_ratio(doc):

    embedded_clauses = 0
    total_sentences = len(doc.sentences)

    for sent in doc.sentences:
        has_embedded_clause = any(is_embedded_clause(token) for token in sent.words)
        if has_embedded_clause:
            embedded_clauses += 1

    
    ratio = embedded_clauses / total_sentences if total_sentences > 0 else 0
    return ratio

def estimated_stressed_syllables(word):

    word = word.lower()
    pronunciations = pronouncing.phones_for_word(word)
    if pronunciations:
        
        pronunciation = pronunciations[0]
        
        stress_count = sum(1 for phoneme in pronunciation.split() if phoneme[-1].isdigit() and phoneme[-1] in '12')
        return stress_count
    else:
        return 0

def ratio_of_stressed_syllables(doc):
    total_syllables = 0
    stressed_syllables = 0

    for sentence in doc.sentences:
        for word in sentence.words:
            word_text = word.text
            syllables = count_syllables(word_text)
            stressed = estimated_stressed_syllables(word_text)
            
            total_syllables += syllables
            stressed_syllables += stressed
    
    return stressed_syllables / total_syllables if total_syllables > 0 else 0.0

def get_stress_pattern(word):
    word = word.lower()
    stress_patterns = []
    
    pronunciations = pronouncing.phones_for_word(word)
    if pronunciations:
        
        pronunciation = pronunciations[0]
        
        for phone in pronunciation.split():
            
            if phone[-1] in '12':  
                stress_patterns.append(1)
            else:
                stress_patterns.append(0)
    
    return stress_patterns

def pairwise_variability_index(doc):


    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words if word.text.isalpha()]
    stress_patterns = []
    
    for word in words:
        stress_patterns.extend(get_stress_pattern(word))


    if len(stress_patterns) < 2:
        return 0.0
    
    
    differences = [abs(stress_patterns[i] - stress_patterns[i + 1]) for i in range(len(stress_patterns) - 1)]
    
    
    average_difference = np.mean(differences)
    
    
    
    
    
    return average_difference

def contains_embedded_clause(sentence):
    
    for word in sentence.words:
        if word.deprel in ['acl', 'advcl', 'ccomp', 'xcomp']:
            return True
    return False

def ratio_of_embedded_clauses(doc):
    sentences = doc.sentences
    
    if not sentences:
        return 0.0
    
    embedded_clause_count = sum(1 for sentence in sentences if contains_embedded_clause(sentence))
    
    ratio = embedded_clause_count / len(sentences)
    
    return ratio

def calculate_noun_overlap(doc):
    overlap_count = 0
    previous_nouns = set()
    
    for sentence in doc.sentences:
        current_nouns = {word.text for word in sentence.words if word.upos == 'NOUN' and word.deprel in {'nsubj', 'dobj', 'nmod'}}
        if previous_nouns & current_nouns:
            overlap_count += 1
        previous_nouns = current_nouns
    
    return overlap_count / len(doc.sentences) if len(doc.sentences) > 0 else 0

def is_rare(word, threshold=0.001):
    
    freq = word_frequency(word, 'en')
    
    return freq < threshold

def rare_words_ratio(doc, threshold=0.001):
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    rare_words = [word for word in words if is_rare(word, threshold)]
    return len(rare_words) / len(words) if words else 0

def zipfian_distribution(x, s):
    return x ** (-s)

def compute_zipfian_loss(doc):
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words if word.text.isalpha()]
    
    if len(words) == 0:
        return np.nan
    
    freq_dist = Counter(words)
    sorted_freqs = sorted(freq_dist.values(), reverse=True)
    
    ranks = np.arange(1, len(sorted_freqs) + 1)
    freqs = np.array(sorted_freqs)
    
    try:
        popt, _ = curve_fit(zipfian_distribution, ranks, freqs, p0=[1.0])
        s = popt[0]
    except Exception as e:
        print(f"Error in fitting Zipfian distribution: {e}")
        return np.nan
    
    fitted_freqs = zipfian_distribution(ranks, s)
    
    
    total_words = len(words)
    freq_norm = freqs / total_words
    fitted_freq_norm = fitted_freqs / total_words
    
    loss_value = np.mean(np.abs(freq_norm - fitted_freq_norm))
    
    return loss_value

def average_text_concreteness(text):
        return avg_text_concreteness(text)

def ratio_concrete_to_abstract(text):
        return concrete_abstract_ratio(text)

def pronoun_sentence_opening_ratio(doc):
    total_sentences = len(doc.sentences)
    initial_pronouns = 0

    for sentence in doc.sentences:
        first_word_pos = sentence.words[0].upos
        if first_word_pos == 'PRON':
            initial_pronouns += 1

    
    ratio = initial_pronouns / total_sentences if total_sentences > 0 else 0
    return ratio

def is_sentence_initial_conjunction(sentence):
    
    if len(sentence.words) > 0:
        first_word_pos = sentence.words[0].pos
        
        if first_word_pos == 'CC':
            return True
    
    return False

def ratio_of_sentence_initial_conjunctions(doc):

    sentences = doc.sentences
    
    if not sentences:
        return 0.0
    
    conjunction_count = sum(1 for sentence in sentences if is_sentence_initial_conjunction(sentence))
    
    ratio = conjunction_count / len(sentences)
    
    return ratio

def summer_index(doc):

    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words if word.upos != "PUNCT"]
    
    
    total_words = len(words)
    if total_words == 0:
        return 0
    
    
    unique_words = len(set(words))
    
    
    ttr = unique_words / total_words
    
    
    total_syllables = 0
    for word in words:
        pronunciations = pronouncing.phones_for_word(word)
        if pronunciations:
            syllable_count = pronouncing.syllable_count(pronunciations[0])
        else:
            
            syllable_count = max(1, len(word) // 3)
        total_syllables += syllable_count
    
    
    avg_syllables_per_word = total_syllables / total_words
    
    
    base_index = 1 / (1 + math.log10(ttr + 1e-10))  
    syllable_factor = math.log(avg_syllables_per_word + 1, 2)  
    enhanced_index = base_index * syllable_factor
    
    return enhanced_index

def is_figurative(sent):

    figurative_clues = {"like", "as", "than", "seems", "appears", "metaphorically", "resembles", "is", "are"}

    for word in sent.words:
        if word.text.lower() in figurative_clues:
            if word.deprel in {"case", "mark"} and word.text.lower() in {"like", "as"}:
                return True
            if word.upos == "AUX" and word.text.lower() in {"is", "are"}:
                for child in sent.words:
                    if child.head == word.id and child.upos in {"NOUN", "PRON"}:
                        return True
    return False

def figurative_vs_literal_ratio(doc):

    figurative_count = sum(is_figurative(sent) for sent in doc.sentences)
    total_sentences = len(doc.sentences)
    
    return figurative_count / total_sentences if total_sentences > 0 else 0.0

def complex_words_rate(doc):
    
    words = [word.text for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    
    complex_words_count = sum(1 for word in words if count_syllables(word) >= 3)
    
    
    complex_words_rate = complex_words_count / total_words
    
    return complex_words_rate

def dale_chall_complex_words_rate(doc):
 
    
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words if word.upos != "PUNCT"]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    
    text = ' '.join(words)
    
    
    dale_chall_complex_words = textstat.difficult_words(text)
    
    
    complex_words_rate = dale_chall_complex_words / total_words
    
    return complex_words_rate

def guirauds_index(doc):
   
    
    
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    
    total_words = len(words)
    
    if total_words == 0:
        return 0
    
    
    unique_words = len(set(words))
    
    
    guirauds_index_value = unique_words / math.sqrt(total_words)
    
    return guirauds_index_value

def sentence_type_ratio(doc):
    
    sentence_type_counts = defaultdict(int)
    total_sentences = len(doc.sentences)
    
    for sentence in doc.sentences:
        
        num_clauses = sum(1 for word in sentence.words if word.deprel in ['ccomp', 'acl', 'advcl'])
        
        if num_clauses > 1:
            
            sentence_type_counts['compound-complex'] += 1
        elif num_clauses == 1:
            
            sentence_type_counts['complex'] += 1
        elif len(sentence.words) > 1 and any(word.deprel == 'conj' for word in sentence.words):
            
            sentence_type_counts['compound'] += 1
        else:
            
            sentence_type_counts['simple'] += 1
    
    
    if total_sentences > 0:
        sentence_type_ratios = {typ: count / total_sentences for typ, count in sentence_type_counts.items()}
    else:
        
        sentence_type_ratios = {typ: 0 for typ in ['simple', 'compound', 'complex', 'compound-complex']}
    
    
    total_ratio = sum(sentence_type_ratios.values())
    if total_ratio > 0:
        sentence_type_ratios = {typ: ratio / total_ratio for typ, ratio in sentence_type_ratios.items()}
    
    return sentence_type_ratios

def calculate_frazier_depth(sentence):

    def build_tree(words):
        tree = {word.id: [] for word in words}
        for word in words:
            if word.head != 0:  
                tree[word.head].append(word.id)
        return tree

    def depth(node_id, tree, current_depth):
        max_depth = current_depth
        for child_id in tree[node_id]:
            max_depth = max(max_depth, depth(child_id, tree, current_depth + 1))
        return max_depth

    
    dependency_tree = build_tree(sentence.words)
    
    
    root_id = next(word.id for word in sentence.words if word.head == 0)
    
    
    return depth(root_id, dependency_tree, 0)

def frazier_depth(doc):

    depths = []
    for sentence in doc.sentences:
        depths.append(calculate_frazier_depth(sentence))
    
    if not depths:
        return 0.0
    
    return sum(depths) / len(depths)

def syll_per_word(doc):

    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    if not words:
        return 0.0

    
    total_syllables = 0
    for word in words:
        syllables = count_syllables(word)
        total_syllables += syllables

    
    syllables_per_word = total_syllables / len(words)
    
    return syllables_per_word

def count_syllables(word):

    word = word.lower()
    pronunciations = pronouncing.phones_for_word(word)
    if pronunciations:
        
        pronunciation = pronunciations[0]
        
        syllables = len(pronunciation.split()) - pronunciation.count('0')
        return syllables
    else:
        return 0

def average_sentence_length(doc):
    
    sentences = [sentence for sentence in doc.sentences]
    
    
    total_sentences = len(sentences)
    
    if total_sentences == 0:
        return 0.0
    
    
    total_words = sum(len(sentence.words) for sentence in sentences)
    
    
    average_length = total_words / total_sentences
    
    return average_length

def compute_average_dependency_distance(doc):
   
    distances = []
    for sentence in doc.sentences:
        
        for word in sentence.words:
            if word.head != 0:  
                head_word = sentence.words[word.head - 1]  
                distance = abs(word.id - head_word.id)  
                distances.append(distance)
    
    if not distances:
        
        return 0
    
    average_distance = np.mean(distances)
    return average_distance

def calculate_cumulative_syntactic_complexity(doc):
    complexity_score = 0
    for sentence in doc.sentences:
        
        complexity_score += sum(1 for word in sentence.words if hasattr(word, 'deprel') and word.deprel in {"acl", "relcl", "ccomp", "xcomp", "nsubj"})
    return complexity_score

def average_syntactic_branching_factor(doc):
    """Calculate the syntactic branching factor of a document."""
    total_branches = 0
    for sentence in doc.sentences:
        
        branches = sum(1 for token in sentence.words if hasattr(token, 'deprel') and token.deprel in {"acl", "relcl", "ccomp", "xcomp", "nsubj"})
        total_branches += branches
    
    branching_factor = total_branches / len(doc.sentences) if len(doc.sentences) > 0 else 0
    return branching_factor

def calculate_structural_complexity_index(doc):
    total_sentences = len(doc.sentences)
    total_clauses = 0
    total_length = 0
    
    for sentence in doc.sentences:
        
        total_clauses += sum(1 for word in sentence.words if hasattr(word, 'deprel') and word.deprel in {"acl", "relcl", "ccomp", "xcomp", "nsubj"})
        total_length += len(sentence.words)
    
    average_sentence_length = total_length / total_sentences if total_sentences > 0 else 0
    average_clauses_per_sentence = total_clauses / total_sentences if total_sentences > 0 else 0
    
    
    
    structural_complexity_index = (average_clauses_per_sentence + average_sentence_length) / 2
    
    return structural_complexity_index

def lexical_overlap(doc):

    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    word_counts = Counter(words)
    overlap = sum(count for count in word_counts.values() if count > 1)
    return overlap / len(words) if words else 0

def compute_yngve_depth(doc):
 
    
    
    yngve_depths = []
    
    for sentence in doc.sentences:
        
        depths = []
        for word in sentence.words:
            depth = 0
            current_word = word
            
            while current_word.head != 0:
                depth += 1
                current_word = sentence.words[current_word.head - 1]
            depths.append(depth)
        
        if depths:
            
            yngve_depth = sum(1 / d for d in depths if d > 0)
            yngve_depths.append(yngve_depth)
    
    if not yngve_depths:
        
        return 0
    
    
    average_yngve_depth = np.mean(yngve_depths)
    
    return average_yngve_depth

def syntactic_branching_factor(sentence):

    
    child_count = {}

    
    for word in sentence.words:
        head = word.head
        if head not in child_count:
            child_count[head] = 0
        child_count[head] += 1

    
    if child_count:
        avg_branching_factor = np.mean(list(child_count.values()))
    else:
        avg_branching_factor = 0.0

    return avg_branching_factor

def branching_factor_for_text(doc):

    branching_factors = [syntactic_branching_factor(sentence) for sentence in doc.sentences]
    
    if branching_factors:
        avg_branching_factor = np.mean(branching_factors)
    else:
        avg_branching_factor = 0.0

    return avg_branching_factor

def frequent_delimiters_rate(doc, delimiters={',', '.', '?', '!'}):

    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return 0.0
    
    delimiter_count = sum(1 for sentence in doc.sentences for token in sentence.tokens if token.text in delimiters)
    
    return delimiter_count / total_words

def lessfrequent_delimiters_rate(doc, delimiters={';', ':'}):
    """
    Computes the rate of less frequent delimiters in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.
        delimiters (set): A set of less frequent delimiter characters to consider.

    Returns:
        float: The rate of less frequent delimiters.
    """
    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return 0.0
    
    delimiter_count = sum(1 for sentence in doc.sentences for token in sentence.tokens if token.text in delimiters)
    
    return delimiter_count / total_words

def parentheticals_and_brackets_rate(doc, delimiters={'(', ')', '[', ']'}):

    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return 0.0
    
    delimiter_count = sum(1 for sentence in doc.sentences for token in sentence.tokens if token.text in delimiters)
    
    return delimiter_count / total_words


    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return 0.0
    
    delimiter_count = sum(1 for sentence in doc.sentences for token in sentence.tokens if token.text in delimiters)
    
    return delimiter_count / total_words

def dashes_and_ellipses_rate(doc, delimiters={'—', '–', '‒', '―', '...', '…'}):
    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return 0.0
    
    delimiter_count = sum(1 for sentence in doc.sentences for token in sentence.tokens if token.text in delimiters)
    
    return delimiter_count / total_words

def compute_hierarchical_structure_complexity(doc):

    depths = []
    
    for sentence in doc.sentences:
        
        for word in sentence.words:
            depth = 0
            current_word = word
            
            while current_word.head != 0:
                depth += 1
                current_word = sentence.words[current_word.head - 1]
            depths.append(depth)
    
    if not depths:
        
        return 0
    
    average_depth = np.mean(depths)
    
    return average_depth

def flesch_reading_ease(text):
        return textstat.flesch_reading_ease(text)

def GFI(text):
        return textstat.gunning_fog(text)

def coleman_liau_index(text):
        return textstat.coleman_liau_index(text)

def ari(text):
        return textstat.automated_readability_index(text)

def dale_chall_readability_score(text):
        return textstat.dale_chall_readability_score(text)

def lix(text):
        return textstat.lix(text)

def smog_index(text):
        return textstat.smog_index(text)

def rix(text):
        return textstat.rix(text)

def nominalization(text):
 
    
    results = getmeasures(text, lang='en')
    
    
    nominalization_score = results['word usage']['nominalization']
    
    
    word_count = len([word for sentence in doc.sentences for word in sentence.words])
    
    
    normalized_score = nominalization_score / word_count if word_count > 0 else 0
    
    return normalized_score














    


    

    


def detailed_conjunctions_usage(doc):
        
    coordinating_conjunctions = {'and', 'or', 'but', 'so', 'for', 'nor', 'yet'}
    subordinating_conjunctions = {'although', 'because', 'since', 'unless', 'while', 'though', 'if', 'as', 'when', 'after', 'before', 'until', 'where', 'whether'}
    correlative_conjunctions = {'either', 'neither', 'not only', 'both', 'whether'}  
    
    
    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return {
            "coordinating": 0.0,
            "subordinating": 0.0,
            "correlative": 0.0
        }
    
    
    coordinating_count = 0
    subordinating_count = 0
    correlative_count = 0

    for sentence in doc.sentences:
        for i, word in enumerate(sentence.words):
            word_lower = word.text.lower()
            if word_lower in coordinating_conjunctions:
                coordinating_count += 1
            elif word_lower in subordinating_conjunctions:
                subordinating_count += 1
            elif word_lower in correlative_conjunctions:
                if word_lower == 'either' and i + 1 < len(sentence.words) and sentence.words[i + 1].text.lower() == 'or':
                    correlative_count += 1
                elif word_lower == 'neither' and i + 1 < len(sentence.words) and sentence.words[i + 1].text.lower() == 'nor':
                    correlative_count += 1
                elif word_lower == 'not' and i + 2 < len(sentence.words) and sentence.words[i + 1].text.lower() == 'only' and sentence.words[i + 2].text.lower() == 'but':
                    correlative_count += 1
                elif word_lower == 'both' and i + 1 < len(sentence.words) and sentence.words[i + 1].text.lower() == 'and':
                    correlative_count += 1
                elif word_lower == 'whether' and i + 1 < len(sentence.words) and sentence.words[i + 1].text.lower() == 'or':
                    correlative_count += 1

    
    coordinating_rate = coordinating_count / total_words
    subordinating_rate = subordinating_count / total_words
    correlative_rate = correlative_count / total_words

    return {
        "coordinating": coordinating_rate,
        "subordinating": subordinating_rate,
        "correlative": correlative_rate
    }

def auxiliary_infipart_modals_usage_rate(doc):
    group1_auxiliaries = {'is', 'are', 'was', 'were', 'do', 'does', 'did'}
    group2_auxiliaries = {'have', 'has', 'had'}
    infinitives_participles = {'be', 'being', 'been'}
    modals = {'can', 'could', 'should', 'will', 'would', 'may', 'might', 'must'}
    
    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return [0.0] * 4
    
    group1_count = sum(1 for sentence in doc.sentences for word in sentence.words if word.text.lower() in group1_auxiliaries)
    group2_count = sum(1 for sentence in doc.sentences for word in sentence.words if word.text.lower() in group2_auxiliaries)
    infinitives_participles_count = sum(1 for sentence in doc.sentences for word in sentence.words if word.text.lower() in infinitives_participles)
    modals_count = sum(1 for sentence in doc.sentences for word in sentence.words if word.text.lower() in modals)
    
    return [
        group1_count / total_words,
        group2_count / total_words,
        infinitives_participles_count / total_words,
        modals_count / total_words
    ]
