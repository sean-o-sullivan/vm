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
    print('The doc is', doc)

    embedding_vector = []

    # Lexical richness measures
    embedding_vector.extend([
        compute_herdans_v(doc),
        compute_brunets_w(doc),
        tuldava_ln_ttr(doc),
        simpsons_index(doc),
        sichel_s_measure(doc),
        orlov_sigma(doc),
        yules_k_characteristic(doc),
        honores_r(doc),
        renyis_entropy(doc),
        hapax_dislegomena_rate(doc),
        dugast_vmi(doc)
    ])

    # syntactic measures
    embedding_vector.extend([
        clauses_per_sentence(doc),
        modifiers_per_noun_phrase(doc),
        coordinated_phrases_per_sentence(doc),
        sentence_length_variation(doc),
        clause_length_variation(doc),
        subordination_index(doc),
        average_sentence_depth(doc),
        compute_average_dependency_distance(doc),
        frazier_depth(doc),
        branching_factor_for_text(doc),
        compute_hierarchical_structure_complexity(doc),
        compute_yngve_depth(doc)
    ])

    # Rhetorical and discourse measures
    embedding_vector.extend([
        ratio_of_passive_voice(doc),
        ratio_of_cleft_sentences(doc),
        calculate_noun_overlap(doc),
        ratio_of_embedded_clauses(doc),
        pronoun_sentence_opening_ratio(doc),
        ratio_of_sentence_initial_conjunctions(doc),
        ratio_of_fronted_adverbials(doc)
    ])

    # Phonological measures
    embedding_vector.extend([
        normalized_assonance(text),
        normalized_alliteration(text),
        tempo_variation(doc),
        rhythmic_complexity(text),
        rhythmic_variability(text)
    ])

    # Lexical concreteness measures
    embedding_vector.extend([
        average_text_concreteness(text),
        ratio_concrete_to_abstract(text),
        figurative_vs_literal_ratio(doc)
    ])

    # Readability indices
    embedding_vector.extend([
        flesch_reading_ease(text),
        GFI(text),
        coleman_liau_index(text),
        ari(text),
        dale_chall_readability_score(text),
        lix(text),
        smog_index(text),
        rix(text)
    ])
    
    return embedding_vector
