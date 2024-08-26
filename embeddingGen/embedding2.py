from stylometricValues import * 
import numpy as np

def generateEmbedding(text):
    """
    Computes a stylometric embedding representation of the text.

    Parameters:
        text (str): The input text sample.

    Returns:
        dict: A dictionary with feature names as keys and corresponding numeric values as values.
    """
    doc = process_text(text)
    embedding_dict = {}

    def add_to_dict(func_name, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            for i, v in enumerate(value):
                embedding_dict[f"{func_name}_{i+1}"] = float(v) if isinstance(v, np.floating) else v
        elif isinstance(value, dict):
            for k, v in value.items():
                embedding_dict[f"{func_name}_{k}"] = float(v) if isinstance(v, np.floating) else v
        else:
            embedding_dict[func_name] = float(value) if isinstance(value, np.floating) else value

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
        compute_inversion_frequencies,
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
        preposition_usage,
        lessfrequent_delimiters_rate,
        parentheticals_and_brackets_rate, 
        calculate_cumulative_syntactic_complexity,
        average_syntactic_branching_factor, 
        calculate_structural_complexity_index,
        lexical_overlap, 
        analyze_passiveness, 
        ratio_of_cleft_sentences,
        calculate_noun_overlap, 
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
        func_name = func.__name__
        if func in [average_text_concreteness, nominalization, ratio_concrete_to_abstract, normalized_assonance, normalized_alliteration, rhythmic_complexity,
                    flesch_reading_ease, GFI, coleman_liau_index, ari,
                    dale_chall_readability_score, lix, smog_index, rix]:
            add_to_dict(func_name, func(text))
        else:
            add_to_dict(func_name, func(doc))

    return embedding_dict
