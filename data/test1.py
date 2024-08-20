import os
import csv
from tqdm import tqdm
import textstat
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_quotation_density(content):
    total_characters = len(content)
    quotation_marks = sum(content.count(q) for q in ("'", '"', "'", "'", """, """, "«", "»", "‹", "›", "「", "」", "『", "』"))
    return quotation_marks / total_characters if total_characters > 0 else 0

def calculate_readability_scores(content):
    
    try:
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(content),
            "ari": textstat.automated_readability_index(content),
            "gunning_fog": textstat.gunning_fog(content),
            "smog_index": textstat.smog_index(content)
        }
    except Exception as e:
        logging.error(f"Error calculating readability scores: {e}")
        return {k: None for k in ["flesch_reading_ease", "ari", "gunning_fog", "smog_index"]}

def process_file(file_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metrics = {"file_path": file_path}
        metrics["quotation_density"] = calculate_quotation_density(content)
        metrics.update(calculate_readability_scores(content))
        return metrics

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

def collect_file_paths(directory):
    
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_paths.append(os.path.join(root, file))
    return file_paths

def write_results_to_csv(results, output_path):
    
    fieldnames = ["file_path", "quotation_density", "flesch_reading_ease", "ari", "gunning_fog", "smog_index"]
    try:
        file_exists = os.path.isfile(output_path)
        with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:  
                writer.writeheader()
            for result in results:
                if result:
                    writer.writerow(result)
        logging.info(f"Metrics written to {output_path}")
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")

def main(input_dir, output_path, batch_size=50):
    
    file_paths = collect_file_paths(input_dir)
    logging.info(f"Found {len(file_paths)} files to process")

    results = []
    for i, file_path in enumerate(tqdm(file_paths, desc="Processing files")):
        result = process_file(file_path)
        if result:
            results.append(result)
        
        
        if (i + 1) % batch_size == 0:
            write_results_to_csv(results, output_path)
            results.clear()  

    
    if results:
        write_results_to_csv(results, output_path)
        results.clear()


bigtext_dir = '/Users/sean/Desktop/vm/datasets/bigText'
output_dir = os.path.dirname(os.path.dirname(bigtext_dir))
output_path = os.path.join(output_dir, 'readability_metrics.csv')


main(bigtext_dir, output_path)
