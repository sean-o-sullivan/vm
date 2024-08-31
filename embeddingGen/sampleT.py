import json
import csv
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def process_jsonl_files(input_jsonl, output_jsonl, output_csv):
    
    input_data = {}
    logging.info(f"Reading input JSONL file: {input_jsonl}")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing input JSONL"):
            entry = json.loads(line)
            custom_id = entry['custom_id']
            input_data[custom_id] = {
                'text': entry['text'],
                'author_id': entry['author_id']
            }

    
    logging.info(f"Processing output JSONL file: {output_jsonl}")
    with open(output_jsonl, 'r', encoding='utf-8') as f, \
         open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['custom_id', 'author_id', 'text'])
        for line in tqdm(f, desc="Processing this output JSONL"):
            entry = json.loads(line)
            custom_id = entry['custom_id']
            response_content = entry['response']['body']['choices'][0]['message']['content']
            if response_content.strip().upper() == 'YES':
                if custom_id in input_data:
                    csv_writer.writerow([
                        custom_id,
                        input_data[custom_id]['author_id'],
                        input_data[custom_id]['text']
                    ])
                else:
                    logging.warning(f"Custom ID {custom_id} not found in input data")

    logging.info(f"Processing complete. results are written to {output_csv}")

if __name__ == "__main__":
    input_jsonl = 'batch_dataset_classification_5K.jsonl'
    output_jsonl = 'batch_dataset_classification_output_5K.jsonl'
    output_csv = 'selected_samples_FromGPTRound2.csv'
    process_jsonl_files(input_jsonl, output_jsonl, output_csv)
