import os
import pandas as pd
import logging
from tqdm import tqdm
import openai
import tiktoken

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["OPENAI_API_KEY"] = "my-key"
openai.api_key = os.getenv("OPENAI_API_KEY")

GENERATE_TEXT_PROMPT = """
You are a versatile writer capable of producing text on various topics. Your task is to write on the given topic:
1. Address the topic thoroughly and thoughtfully
2. Use an appropriate tone and style for the subject matter
3. Aim for the specified approximate token count
4. Do not include a title or any introductory text
Your goal is to produce engaging and informative content on the provided topic.
"""

def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def extract_topic(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract a general topic from the given text in 5 words or less."},
            {"role": "user", "content": f"Extract the main topic from this text: {text[:500]}"}
        ]
    )
    topic = response['choices'][0]['message']['content']
    return topic

def generate_text_on_topic(topic, token_count):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": GENERATE_TEXT_PROMPT},
            {"role": "user", "content": f"Write about {topic} for approximately {token_count} tokens."}
        ]
    )
    generated_text = response['choices'][0]['message']['content']
    return generated_text

def remove_delimiters(text):
    return text.replace("#/#\\#|||#/#\\#|||#/#\\#", "")

def process_samples(input_csv, output_csv):
    logging.info(f"Loading dataset from: {input_csv}")
    
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    if df.empty:
        logging.error(f"No entries found in the input CSV file: {input_csv}. Exiting.")
        return

    logging.info(f"Total samples to process: {len(df)}")

    results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        author = row['author']
        processed_sample = row['cleaned_text']
        cleaned_sample = remove_delimiters(processed_sample)
        original_token_count = count_tokens(cleaned_sample)
        topic = extract_topic(cleaned_sample)
        generated_text = generate_text_on_topic(topic, original_token_count)

        results.append({
            "author": author,
            "original_text": cleaned_sample[:500],
            "extracted_topic": topic,
            "original_token_count": original_token_count,
            "generated_text": generated_text[:500],
            "generated_token_count": count_tokens(generated_text)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    logging.info(f"Generated samples saved to: {output_csv}")

if __name__ == "__main__":

    input_csvs = ['ABB_30.csv', 'AGG_30.csv']
    for input_csv in input_csvs:
        output_csv = f'topic_based_samples_{input_csv.split(".")[0]}.csv'
        logging.info(f"Starting the processing for {input_csv}...")
        process_samples(input_csv, output_csv)
        logging.info(f"Processing completed for {input_csv}. Generated samples saved to {output_csv}.")
