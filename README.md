# Important Links

## Data Pre-Processing

- **Dataset Selection and Filtering:**
  - **Author Filtering**: 
    - [Select 1K Authors with >5 Samples](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/Select1K.py)
  - **Batch Requests**:
    - [Batch Request Script](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/run3_part3-To%20make%20batch%20request.py)
    - [Batch Request Placement](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/batching.ipynb)
  - **Batch Processing**:
    - [Received Batch Output (5K)](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/batch_dataset_classification_output_5K.jsonl)
    - [Suitability Processing](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/isSuitableForAuthorship.py)

- **Text Cleaning Functions:**
  - Citations and Tables:
    - [cleam3.py](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/gold_data/cleam3.py)
    - [CorpusSwipe.py](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/gold_data/pro/CorpusSwipe.py)
  - BAWE Specific:
    - [removeWords.py](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/gold_data/pro/archive/removeWords.py)

- **Cleaning Statistics:**
  - [Archive for Cleaning Statistics](https://github.com/sean-o-sullivan/vm/tree/main/embeddingGen/gold_data/archive3)

- **Corpora:**
  - Raw Corpora:
    - [Final Raw Data](https://github.com/sean-o-sullivan/vm/tree/main/embeddingGen/gold_data/final_raw)
  - Cleaned Corpora:
    - [Gutenberg Cleaned](https://media.githubusercontent.com/media/sean-o-sullivan/vm/refs/heads/main/embeddingGen/gold_data/pro/Gutenberg_Clean2.csv)
    - [BAWE Cleaned](https://media.githubusercontent.com/media/sean-o-sullivan/vm/refs/heads/main/embeddingGen/gold_data/pro/archive/BAWE_Clean.csv)

- **Feature Definitions and Embedding Generation:**
  - [Stylometric-Embedding Feature Definitions](https://github.com/sean-o-sullivan/vm/blob/clean/embeddingGen/working/stylometricValues.py)
  - [Embedding Generation Function](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embedding2.py)

- **Z-Score and Corpus Statistics:**
  - Z-Score Calculation and Violin Plot Script:
    - [ee.py](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/ee.py)
  - Calculated Statistics:
    - [Embedding Stats CSV](https://media.githubusercontent.com/media/sean-o-sullivan/vm/refs/heads/main/embeddingGen/working/embeddings/embedding_stats2.csv)
  - Percentile Normalization (for newly generated embeddings):
    - [Z-Score Percentile Normalization](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/ii.py)

## Model Training and Evaluation

- **Triplet Dataset Generation:**
  - Virtual-Capping Sampling:
    - [triplets2.py](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/normalisedandready/triplets2.py)
  - Final Version:
    - [triplets_USE.py](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/normalisedandready/triplets_USE.py)

- **Training/Evaluation Scripts:**
  - [opto.py](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/normalisedandready/opto.py)
  - [m13p.py](https://github.com/sean-o-sullivan/vm/commits/main/embeddingGen/working/embeddings/normalisedandready/m13p.py) (Previous iterations can be found in the directory, labeled `m*.py`)

- **Final Training Data:**
  - **BAWE and Gutenberg Datasets**:
    - All samples follow this structure: `embeddingGen/working/embeddings/normalisedandready/Final-Triplets_X_Y_|Z|_VTLA_CB.csv`, where:
      - **X** is either **B** or **G** (BAWE or Gutenberg)
      - **Y** is either **30** or **70** (evaluation or training set)
      - **Z** is a duplicate file to prevent overwriting
      - **A** is the Virtual Text Limit used
      - **B** is the context size used
- Final Training Data was a combination of VTL5, C3 of both complete corpora.
    - BAWE Files:
      - [BAWE 30% Triplets](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/normalisedandready/Final-Triplets_B_30_%7C_VTL5_C3.csv)
      - [BAWE 70% Triplets](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/normalisedandready/Final-Triplets_B_70_%7C_VTL5_C3.csv)
    - Gutenberg Files:
      - [Gutenberg 30% Triplets](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/normalisedandready/Final-Triplets_G_30_%7C_VTL5_C3.csv)
      - [Gutenberg 70% Triplets](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/normalisedandready/Final-Triplets_G_70_%7C_VTL5_C3.csv)
  - **Normalised Adversarial GPT Embeddings**:
    - [Adversarial Embeddings CSVs](https://github.com/sean-o-sullivan/vm/tree/main/embeddingGen/working/embeddings/normalized_adversarial_csvs)

## Interpretability

- **Embedding Space Visualization:**
  - [vis.py](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/normalisedandready/vis.py)

- **Metadata:**
  - [Gutenberg Sample Metadata (Writing Topics)](https://github.com/sean-o-sullivan/vm/blob/main/embeddingGen/working/embeddings/normalisedandready/GG_100_updated_core_info_only.csv)


## Evaluation and Testing

- **GPT-4o Data Testing**:
  - **Manual Testing on AI Content Detectors**:
    - We manually tested the new GPT-4o data on AI content detectors, covering a subset of the data due to time constraints. Results can be found here:
      - [Testing Spreadsheet](https://docs.google.com/spreadsheets/d/1ZM_vkZbx43YjG5FArvqxJEE3IDRYYAafqJTF00-ei1Y/edit?gid=1317019956#gid=1317019956)
  - **Handling GPT Refusals**:
    - Wherever GPT refused the GPT-4o samples, they were subsequently removed from the evaluation datasets:
      - [Refusal Tracking and Removed Samples](https://docs.google.com/spreadsheets/d/1F3phebzd_P-WF-IJTePCnxkcHjzfznhMZCqaiEQ0Q2w/edit?gid=0#gid=0)

- **Negative Class Evaluation for GPT Adversarial Cases**:
  - **Recorded Evaluation Data**:
    - This spreadsheet contains the complete recorded evaluation data against GPT adversarial cases, categorized by context size and dataset (BAWE or Gutenberg). Each sheet is labeled as CX_BB/GG, where:
      - **CX** is the context size,
      - **BB** is BAWE,
      - **GG** is Gutenberg.
    - As you scroll through each sheet, you’ll find metrics for each testing case. Note: Since this was negative class evaluation only, model accuracies for each corpus and context size were manually recorded:
      - [Evaluation Data Against Adversarial Cases](https://docs.google.com/spreadsheets/d/1_VHQWRCBHYe9eiX1raw0LPDY8jWLprW3wL2TmcXRWKs/edit?gid=67857661#gid=67857661)