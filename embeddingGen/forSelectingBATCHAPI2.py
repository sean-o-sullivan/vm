import os
import json
import pandas as pd
import logging
from tqdm import tqdm

# System prompt for GPT-4
SYSTEM_PROMPT = """
Your sole purpose is to critically evaluate the provided text for authorship verification suitability. Reject any text with dialogue, incoherence, poor structure, or inconsistencies. Accept only texts with clear, consistent, and unique authorial voice. Output 'YES' if suitable, otherwise output 'NO'.
"""

def remove_delimiters(text):
    """Removes custom text delimiters from the processed sample."""
    return text.replace("#/#\\#|||#/#\\#|||#/#\\#", "")

def create_batch_request(custom_id, content):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-08-06",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            "max_tokens": 5  # Lowered to 5 to minimize cost
        }
    }

def process_samples(input_csv, output_file):
    logging.info(f"Loading dataset from: {input_csv}")
    
    # Load the selected samples CSV
    df = pd.read_csv(input_csv)
    
    if df.empty:
        logging.error("No entries found in the selected_samples.csv. Exiting.")
        return

    logging.info(f"Total samples to process: {len(df)}")

    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            author = row['author']
            book = row['book']
            sample_id = row['sample_id']
            processed_sample = row['processed_sample']

            # Remove text delimiters from the processed sample
            cleaned_sample = remove_delimiters(processed_sample)
            custom_id = f"{author}-{book}-{sample_id}"
            batch_request = create_batch_request(custom_id, cleaned_sample)
            json.dump(batch_request, jsonl_file)
            jsonl_file.write('\n')

    logging.info(f"Batch dataset created: {output_file}")

if __name__ == "__main__":
    input_csv = 'selected_samples.csv'
    output_file = 'batch_dataset_classification_5K.jsonl'
    
    logging.info("Starting the batch request generation process...")
    process_samples(input_csv, output_file)
    logging.info("Batch request generation completed.")
