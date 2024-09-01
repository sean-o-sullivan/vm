import pandas as pd
from tqdm import tqdm
import openai
import os

os.environ["OPENAI_API_KEY"] = "my-key"
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_question_for_gpt(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Your task is to generate a question that asks for a piece of writing similar to the provided text, including the general idea of the topic, format, and only use general descriptive words for how long the text should be, do not ever mention approximate values. The author will not be provided this text, so do not mention it."},
            {"role": "user", "content": f"Based on this provided text, please write a question. DO NOT MENTION THE PROVIDED TEXT in your question: \"{text}\""}
        ]
    )
    question = response['choices'][0]['message']['content']
    return question

def generate_mimicry(author_text, question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Your task is to generate a piece of text that mimics the writing style of a given author. You will be provided with a sample of the author's past writing, which you should use as a reference for style, tone, and typical word choice. Following the sample, you will be given a specific topic to write about in the form of a question. Your response should emulate the author's style while focusing on the new topic. Aim for a response that is approximately 600 tokens in length. Do not include a title."},
            {"role": "user", "content": f"Here is a sample of the author's past writing: {author_text}"},
            {"role": "user", "content": f"Based on and mimicking this author's style, please respond to the following question: {question}"}
        ]
    )
    generated_text = response['choices'][0]['message']['content']
    return generated_text

def process_author_samples(author_id, samples):
    results = []
    for sample in samples:
        excerpt = sample[:100]  
        question = create_question_for_gpt(sample)
        context_piece = sample  
        generated_text = generate_mimicry(context_piece, question)
        results.append({
            'author_id': author_id,
            'original_excerpt': excerpt,
            'context_piece': context_piece,
            'generated_sample': generated_text
        })
    return results

def generate_for_all_authors(df, X):
    all_results = []
    for author_id, group in df.groupby('author'):
        if len(group) >= 2:  
            samples = group['cleaned_text'].sample(n=min(X, len(group)), random_state=0).tolist()
            author_results = process_author_samples(author_id, samples)
            all_results.extend(author_results)
    return all_results

def save_results(results, output_file):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def process_files(file_names, X):
    for file_name in file_names:
        print(f"Processing {file_name}...")
        df = pd.read_csv(file_name)
        results = generate_for_all_authors(df, X)
        output_file = f"generated_samples_{file_name}"
        save_results(results, output_file)
        print(f"Results saved to {output_file}")

def main():
    file_names = ['ABB_30.csv', 'AGG_30.csv']
    X = 5  # context
    process_files(file_names, X)

if __name__ == "__main__":
    main()
