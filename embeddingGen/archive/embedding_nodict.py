from stylometricValues import * 
import numpy as np  

def generateEmbedding(text):
    """
    Computes a stylometric embedding representation of the text.

    Parameters:
        text (str): The input text sample.

    Returns:
        list: A flat list of numeric values representing the stylometric embedding of the text.
    """
    doc = process_text(text)
    embedding_vector = []

    def extend_embedding(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            embedding_vector.extend(value)
        elif isinstance(value, dict):
            embedding_vector.extend(value.values())
        else:
            embedding_vector.append(value)

    # Apply all feature extraction functions
    feature_functions = [
        compute_herdans_v, 
        compute_brunets_w, 
        tuldava_ln_ttr, 
        simpsons_index,
        sichel_s_measure, 
        orlov_sigma, 
        pos_frequencies, 
        yules_k_characteristic,
        honores_r, 
        renyis_entropy, 
        perplexity, 
        burstiness, 
        hapax_dislegomena_rate,
        dugast_vmi, 
        average_word_length, 
        prosodic_patterns, 
        ratio_of_inverted_structures,
        clauses_per_sentence,
        modifiers_per_noun_phrase, 
        coordinated_phrases_per_sentence,
        coordinate_phrases_ratio, 
        sentence_length_variation, 
        clause_length_variation,
        dependent_clauses_ratio, 
        subordination_index, 
        average_sentence_depth,
        compute_average_dependency_distance, 
        frazier_depth, 
        branching_factor_for_text,
        compute_hierarchical_structure_complexity, 
        compute_yngve_depth, long_word_rate,
        short_word_rate, 
        lexical_density, 
        calculate_embedded_clause_ratio,
        ratio_of_stressed_syllables,
        pairwise_variability_index, 
        compute_zipfian_loss,
        frequent_delimiters_rate, 
        lessfrequent_delimiters_rate,
        parentheticals_and_brackets_rate, 
        calculate_cumulative_syntactic_complexity,
        average_syntactic_branching_factor, 
        calculate_structural_complexity_index,
        lexical_overlap, ratio_of_passive_voice, 
        ratio_of_cleft_sentences,
        calculate_noun_overlap, 
        ratio_of_embedded_clauses, 
        pronoun_sentence_opening_ratio,
        ratio_of_sentence_initial_conjunctions, 
        calculate_fronted_adverbial_ratio,
        rare_words_ratio, summer_index, 
        dale_chall_complex_words_rate, 
        guirauds_index,
        syll_per_word, 
        average_sentence_length, 
        normalized_assonance, 
        normalized_alliteration,
        tempo_variation, 
        rhythmic_complexity, 
        complex_words_rate, 
        detailed_conjunctions_usage,
        auxiliary_infipart_modals_usage_rate, 
        sentence_type_ratio, 
        figurative_vs_literal_ratio, 
        flesch_reading_ease,
        GFI, 
        coleman_liau_index, 
        ari, 
        dale_chall_readability_score, 
        lix, 
        smog_index,
        rix, 
    ]

    for func in feature_functions:
        if func in [average_text_concreteness, nominalization, ratio_concrete_to_abstract, normalized_assonance, normalized_alliteration, rhythmic_complexity,
                    flesch_reading_ease, GFI, coleman_liau_index, ari,
                    dale_chall_readability_score, lix, smog_index, rix]:
            extend_embedding(func(text))
        else:
            extend_embedding(func(doc))

    # Convert any np.float64 or similar to standard Python float
    embedding_vector = [float(x) if isinstance(x, np.floating) else x for x in embedding_vector]

    return embedding_vector
