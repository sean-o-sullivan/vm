import os
import json
import pandas as pd
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)

#script made for adding the metadata to the samples we see in the visualisation plot, the system prompt was claude
SYSTEM_PROMPT = """
You are an expert literary classifier specializing in texts from the Project Gutenberg corpus. Your task is to read the provided text sample and accurately determine its primary domain of writing. Use the following categories and subcategories:

---

**Categories and Subcategories:**

1. **Fiction**
   - **Novel**
     - Literary Fiction
     - Science Fiction
     - Fantasy
     - Mystery/Detective
     - Romance
     - Horror
     - Adventure
     - Historical Fiction
     - Satire
     - Dystopian
   - **Short Story**
   - **Children's Literature**
   - **Fable/Folk Tale**
   - **Mythology**

2. **Non-Fiction**
   - **Biography/Autobiography/Memoir**
     - Political Figures
     - Historical Figures
     - Scientists
     - Artists
     - Personal Memoirs
   - **History**
     - Ancient History
     - Medieval History
     - Modern History
     - Military History
     - Cultural History
     - Social History
     - Economic History
     - History of Science
     - World History
     - Regional History
   - **Science**
     - Physics
     - Chemistry
     - Biology
     - Astronomy
     - Geology
     - Mathematics
     - Medicine
     - Environmental Science
     - Engineering
     - Technology
     - Computer Science
   - **Philosophy**
     - Metaphysics
     - Epistemology
     - Ethics
     - Logic
     - Aesthetics
     - Political Philosophy
     - Philosophy of Science
     - Existentialism
     - Eastern Philosophy
   - **Religion/Spirituality**
     - Comparative Religion
     - Theology
     - Spiritual Writings
     - Religious History
     - Mythology
   - **Travelogue**
     - Exploration
     - Travel Narratives
     - Guidebooks
     - Adventure Travel
     - Cultural Observations
   - **Essays**
     - Personal Essays
     - Critical Essays
     - Academic Essays
     - Social Commentary
     - Philosophical Essays
   - **Self-Help/Instructional**
     - Personal Development
     - Educational Materials
     - Language Learning
     - How-To Guides
     - Health and Wellness
     - Business and Management
     - Parenting
   - **Politics/Economics**
     - Political Theory
     - Economic Theory
     - Political Science
     - Public Policy
     - International Relations
     - Social Justice
     - Government Studies
   - **Letters/Correspondence**
     - Personal Letters
     - Official Correspondence
     - Open Letters
     - Epistolary Collections
   - **Art/Culture**
     - Art History
     - Music Theory
     - Literary Criticism
     - Cultural Studies
     - Architecture
     - Theatre Studies
     - Film Studies
   - **Language/Literature Studies**
     - Literary Criticism
     - Linguistics
     - Grammar
     - Rhetoric
     - Comparative Literature
   - **Law**
     - Legal Treatises
     - Case Law
     - Legal Philosophy
     - Criminal Law
     - Civil Law
     - International Law
   - **Social Sciences**
     - Sociology
     - Anthropology
     - Psychology
     - Education
     - Geography
     - Archaeology
     - Gender Studies
     - Ethnic Studies
     - Social Work
   - **Science and Technology**
     - Technological Developments
     - Technical Manuals
     - Engineering Texts
     - Scientific Research
     - Innovation Studies
   - **Education**
     - Pedagogy
     - Curriculum Studies
     - Educational Psychology
     - Teaching Methods
     - Educational Policy
   - **Medicine/Health**
     - Medical Texts
     - Public Health
     - Anatomy
     - Physiology
     - Nutrition
     - Mental Health
   - **Business/Finance**
     - Entrepreneurship
     - Economics
     - Management
     - Accounting
     - Marketing
     - Investment
   - **Agriculture**
     - Farming Techniques
     - Agricultural Science
     - Horticulture
     - Animal Husbandry
   - **Miscellaneous Non-Fiction**
     - Any other non-fiction work not covered above

3. **Poetry**
   - **Epic Poetry**
   - **Lyric Poetry**
   - **Narrative Poetry**
   - **Sonnet**
   - **Free Verse**
   - **Haiku**
   - **Ode**
   - **Elegy**

4. **Drama**
   - **Tragedy**
   - **Comedy**
   - **Historical Play**
   - **Melodrama**
   - **Farce**

5. **Reference**
   - **Dictionary**
   - **Encyclopedia**
   - **Manual/Handbook**
   - **Academic Textbook**
   - **Guidebook**
   - **Almanac**
   - **Atlas**

6. **Religious Texts**
   - **Sacred Scriptures**
   - **Sermons**
   - **Hymns**
   - **Prayers**
   - **Religious Commentary**
   - **Devotional Works**

7. **Speeches**
   - **Political Speeches**
   - **Inaugural Addresses**
   - **Orations**
   - **Commencement Addresses**
   - **Sermons**

8. **Legal Documents**
   - **Constitutions**
   - **Treaties**
   - **Legislation**
   - **Declarations**
   - **Court Opinions**

9. **Periodicals**
   - **Newspaper Articles**
   - **Magazines**
   - **Journals**
   - **Academic Journals**
   - **Bulletins**

10. **Miscellaneous**
    - **Diaries/Journals**
    - **Recipes/Cookbooks**
    - **Technical Manuals**
    - **Speeches**
    - **Other** (if none of the above categories apply)

---

**Instructions:**

- **Read the entire text sample thoroughly.**
- **Determine the most appropriate category and subcategory that best describes the text.**
- **If the text fits multiple categories, choose the one that best represents the primary content.**
- **Output format:** `"Category - Subcategory - Specific Subcategory"` (e.g., `Non-Fiction - Science - Biology`)
- **Do not include any additional text, explanations, or comments in your output.**
- **If the text does not fit any category, output:** `"Miscellaneous - Other"`

---

**Examples:**

- **Sample Text:** "The study of celestial objects has fascinated humans for millennia..."
  - **Output:** `Non-Fiction - Science - Astronomy`

- **Sample Text:** "An Inquiry into the Nature and Causes of the Wealth of Nations..."
  - **Output:** `Non-Fiction - Politics/Economics - Economic Theory`

- **Sample Text:** "This book provides a comprehensive guide to the principles of organic chemistry..."
  - **Output:** `Non-Fiction - Science - Chemistry`

- **Sample Text:** "I wandered lonely as a cloud that floats on high o'er vales and hills..."
  - **Output:** `Poetry - Lyric Poetry`

- **Sample Text:** "The Republic is a Socratic dialogue authored by Plato around 375 BC..."
  - **Output:** `Non-Fiction - Philosophy - Political Philosophy`

- **Sample Text:** "This manual covers the installation and maintenance of electrical systems..."
  - **Output:** `Non-Fiction - Science and Technology - Technical Manuals`

---

**Additional Guidelines:**

- **Consistency is Key:** Ensure your categorization is consistent across similar texts.
- **Historical Context:** Consider the historical period of the text if it helps in categorization.
- **Genre Blending:** If a text blends genres, focus on the predominant one.
- **Avoid Assumptions:** Base your classification solely on the provided text.
- **Specificity:** Choose the most specific subcategory applicable.

"""

def remove_delimiters(text):
    """Removes custom text delimiters from the processed sample."""
    return text.replace("#/#\\#|||#/#\\#|||#/#\\#", "")

def create_batch_request(custom_id, content):
    """Creates a batch request for the GPT-4 API."""
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
            "max_tokens": 30,  
            "temperature": 0, 
            "n": 1,            
            "stop": None      
        }
    }

def process_samples(input_csv, output_file):
    """Processes samples from the CSV and creates batch requests."""
    logging.info(f"Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
   
    if df.empty:
        logging.error("No entries found in the selected_samples.csv. Exiting.")
        return

    logging.info(f"Total samples to process: {len(df)}")

    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            # author = row['author']
            book = row['text']
            sample_id = row['custom_id']
            batch_request = create_batch_request(sample_id, book)
            json.dump(batch_request, jsonl_file)
            jsonl_file.write('\n')

    logging.info(f"Batch dataset created: {output_file}")

if __name__ == "__main__":
    input_csv = '/home/aiadmin/Desktop/code/vm/embeddingGen/selected_samples_FromGPTRound2.csv'
    output_file = 'batch_dataset_classification_topiczF.jsonl'
   
    logging.info("Starting the batch request generation process...")
    process_samples(input_csv, output_file)
    logging.info("Batch request generation completed.")