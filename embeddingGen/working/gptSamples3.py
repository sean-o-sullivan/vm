import os
import pandas as pd
import logging
from tqdm import tqdm
from openai import OpenAI
import tiktoken
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["OPENAI_API_KEY"] = "my-key"
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

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

def create_question_for_gpt(text):
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
    return text.replace("'''", "")

def process_samples(input_csv, output_mimicry_csv, output_topic_csv, max_samples_per_author):
    logging.info(f"Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if df.empty:
        logging.error(f"No entries found in the input CSV file: {input_csv}. Exiting.")
        return

    logging.info(f"Total samples to process: {len(df)}")

    mimicry_results = []
    topic_results = []
    author_sample_count = defaultdict(int)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        author = row['author']
        
        if author_sample_count[author] >= max_samples_per_author:
            continue
        
        processed_sample = row['cleaned_text']
        cleaned_sample = remove_delimiters(processed_sample)
        original_token_count = count_tokens(cleaned_sample)
        question = create_question_for_gpt(cleaned_sample)
        mimicry_sample = generate_mimicry(cleaned_sample, question)

        mimicry_results.append({
            "author": author,
            "original_text": cleaned_sample[:500],
            "past_example": cleaned_sample[:500],
            "generated_question": question,
            "generated_mimicry": mimicry_sample[:500]
        })
        topic = extract_topic(cleaned_sample)
        topic_based_sample = generate_text_on_topic(topic, original_token_count)

        topic_results.append({
            "author": author,
            "original_text": cleaned_sample[:500],
            "extracted_topic": topic,
            "original_token_count": original_token_count,
            "generated_text": topic_based_sample[:500],
            "generated_token_count": count_tokens(topic_based_sample)
        })
        author_sample_count[author] += 1

    mimicry_df = pd.DataFrame(mimicry_results)
    mimicry_df.to_csv(output_mimicry_csv, index=False)
    logging.info(f"Mimicry samples saved to: {output_mimicry_csv}")
    topic_df = pd.DataFrame(topic_results)
    topic_df.to_csv(output_topic_csv, index=False)
    logging.info(f"Topic-based samples saved to: {output_topic_csv}")

if __name__ == "__main__":
    input_csvs = ['ABB_30.csv', 'AGG_30.csv']
    max_samples_per_author = 2

    for input_csv in input_csvs:
        output_mimicry_csv = f'mimicry_samples_GPT3{input_csv.split(".")[0]}.csv'
        output_topic_csv = f'topic_based_samples_GPT3{input_csv.split(".")[0]}.csv'
        logging.info(f"Starting the processing for {input_csv}...")
        process_samples(input_csv, output_mimicry_csv, output_topic_csv, max_samples_per_author)
        logging.info(f"Processing completed for {input_csv}. Outputs saved to {output_mimicry_csv} and {output_topic_csv}.")
