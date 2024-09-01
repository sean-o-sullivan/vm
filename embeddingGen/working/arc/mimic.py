import os
import pandas as pd
import logging
from tqdm import tqdm
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["OPENAI_API_KEY"] = "my-key"
openai.api_key = os.getenv("OPENAI_API_KEY")

CREATE_QUESTION_PROMPT = """
Generate a thought-provoking question that encourages a response in the style and on a topic similar to the given text. 
The question should:
1. Not directly reference the provided text
2. Be open-ended and stimulate creative thinking
3. Relate to themes, tone, or subject matter present in the sample
4. Avoid specifying word counts or length requirements
Your task is to craft a question that would naturally lead to a response mimicking the author's style and subject matter.
"""

GENERATE_MIMICRY_PROMPT = """
You are an expert in literary style mimicry. Your task is to create a text that closely imitates the author's unique style based on the provided sample. 
Follow these guidelines:
1. Analyze the author's voice, tone, sentence structure, vocabulary, and thematic elements
2. Craft a response to the given question that embodies these stylistic elements
3. Maintain consistency with the author's typical subject matter and perspective
4. Do not include a title or any introductory text
5. Aim to make the mimicry as convincing and authentic as possible
Your goal is to produce text that could plausibly be mistaken for the original author's work.
"""

def create_question_for_gpt(text):
    """Generate a question based on the provided text."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": CREATE_QUESTION_PROMPT},
            {"role": "user", "content": f"Generate a question based on this text: \"{text[:500]}\""}
        ]
    )
    question = response['choices'][0]['message']['content']
    return question

def generate_mimicry(author_text, question):
    """Generate a text that mimics the author's style."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": GENERATE_MIMICRY_PROMPT},
            {"role": "user", "content": f"Author's sample: {author_text[:500]}"},
            {"role": "user", "content": f"Question to respond to: {question}"}
        ]
    )
    generated_text = response['choices'][0]['message']['content']
    return generated_text

def remove_delimiters(text):
    """Removes custom text delimiters from the processed sample."""
    return text.replace("#/#\\#|||#/#\\#|||#/#\\#", "")

def process_samples(input_csv, output_csv):
    """Processes samples from the CSV and generates questions and mimicry."""
    logging.info(f"Loading dataset from: {input_csv}")
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
        question = create_question_for_gpt(cleaned_sample)
        generated_sample = generate_mimicry(cleaned_sample, question)

        results.append({
            "author": author,
            "original_text": cleaned_sample[:500],
            "past_example": cleaned_sample[:500],
            "generated_question": question,
            "generated_mimicry": generated_sample[:500]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    logging.info(f"Mimicry samples saved to: {output_csv}")

if __name__ == "__main__":
    input_csvs = ['ABB_30.csv', 'AGG_30.csv']

    for input_csv in input_csvs:
        output_csv = f'mimicry_samples_{input_csv.split(".")[0]}.csv'
        logging.info(f"Starting the processing for {input_csv}...")
        process_samples(input_csv, output_csv)
        logging.info(f"Processing completed for {input_csv}. Mimicry samples saved to {output_csv}.")
