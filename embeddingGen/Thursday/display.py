import json

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def main():
    input_file = '../batch_dataset_classification.jsonl'
    output_file = '../batch_dataset_classification_output.jsonl'

    input_data = read_jsonl(input_file)
    output_data = read_jsonl(output_file)

    yes_entries = []

    for input_entry, output_entry in zip(input_data, output_data):
        if output_entry['response']['body']['choices'][0]['message']['content'] == 'YES':
            yes_entries.append(input_entry)

    if not yes_entries:
        print("No entries with 'YES' responses found.")
        return

    for entry in yes_entries:
        print(f"Custom ID: {entry['custom_id']}")
        print("Content:")
        print(entry['body']['messages'][1]['content'])
        print("\n" + "="*50 + "\n")
        input("Press Enter to see the next entry...")

if __name__ == "__main__":
    main()
