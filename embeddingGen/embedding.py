from stylometricValues import *  

def generateEmbedding(text):
    """
    Computes a stylometric embedding representation of the text.

    Parameters:
        text (str): The input text sample.

    Returns:
        list: A list representing the stylometric embedding of the text.
    """
    
    doc = process_text(text)
    
    
    herdans_v = compute_herdans_v(doc)
    brunets_w = compute_brunets_w(doc)
    tuldava_ln_ttr = tuldava_ln_ttr(doc)
    simpsons_idx = simpsons_index(doc)
    sichel_s = sichel_s_measure(doc)
    orlov_sigma = orlov_sigma(doc)
    yules_k = yules_k_characteristic(doc)
    honores_r = honores_r(doc)
    renyis_entropy = renyis_entropy(doc)
    hapax_dislegomena = hapax_dislegomena_rate(doc)
    dugast_vmi = dugast_vmi(doc)
    
    
    clauses_per_sentence = clauses_per_sentence(doc)
    modifiers_per_noun = modifiers_per_noun_phrase(doc)
    coordinated_phrases_per_sentence = coordinated_phrases_per_sentence(doc)
    sentence_length_variation = sentence_length_variation(doc)
    clause_length_variation = clause_length_variation(doc)
    subordination_idx = subordination_index(doc)
    avg_sentence_depth = average_sentence_depth(doc)
    avg_dependency_distance = compute_average_dependency_distance(doc)
    frazier_depth_value = frazier_depth(doc)
    syntactic_branching_factor = branching_factor_for_text(doc)
    hierarchical_structure_complexity = compute_hierarchical_structure_complexity(doc)
    yngve_depth = compute_yngve_depth(doc)
    
    
    ratio_passive_voice = ratio_of_passive_voice(doc)
    ratio_cleft_sentences = ratio_of_cleft_sentences(doc)
    noun_overlap = calculate_noun_overlap(doc)
    ratio_embedded_clauses = ratio_of_embedded_clauses(doc)
    pronoun_sentence_opening_ratio = pronoun_sentence_opening_ratio(doc)
    ratio_initial_conjunctions = ratio_of_sentence_initial_conjunctions(doc)
    ratio_fronted_adverbials = ratio_of_fronted_adverbials(doc)
    
    
    assonance = normalized_assonance(text)
    alliteration = normalized_alliteration(text)
    tempo_variation = tempo_variation(doc)
    rhythmic_complexity = rhythmic_complexity(text)
    rhythmic_variability = rhythmic_variability(text)
    
    
    avg_concreteness = average_text_concreteness(text)
    concrete_to_abstract_ratio = ratio_concrete_to_abstract(text)
    
    
    flesch = flesch_reading_ease(text)
    gfi = GFI(text)
    coleman_liau = coleman_liau_index(text)
    ari_index = ari(text)
    dale_chall_score = dale_chall_readability_score(text)
    lix_score = lix(text)
    smog = smog_index(text)
    rix_score = rix(text)
    
    
    embedding_vector = [
        herdans_v, brunets_w, tuldava_ln_ttr, simpsons_idx, sichel_s, orlov_sigma, yules_k, honores_r, 
        renyis_entropy, hapax_dislegomena, dugast_vmi, clauses_per_sentence, modifiers_per_noun, 
        coordinated_phrases_per_sentence, sentence_length_variation, clause_length_variation, 
        subordination_idx, avg_sentence_depth, avg_dependency_distance, frazier_depth_value, 
        syntactic_branching_factor, hierarchical_structure_complexity, yngve_depth, ratio_passive_voice, 
        ratio_cleft_sentences, noun_overlap, ratio_embedded_clauses, pronoun_sentence_opening_ratio, 
        ratio_initial_conjunctions, ratio_fronted_adverbials, assonance, alliteration, tempo_variation, 
        rhythmic_complexity, rhythmic_variability, avg_concreteness, concrete_to_abstract_ratio, flesch, 
        gfi, coleman_liau, ari_index, dale_chall_score, lix_score, smog, rix_score
    ]
    
    return embedding_vector
