import pandas as pd
from tqdm.auto import tqdm
import ast
import itertools
from math import comb
from random import sample
import numpy as np

def generate_all_true_pairs(df, virtual_text_limit=20):
    max_pairs_limit = comb(virtual_text_limit, 1) * (virtual_text_limit - 1)

    all_true_pairs = []
    all_context_embeddings_with_index = []
    context_index = 0

    for author_id, group in tqdm(df.groupby('author'), desc='Processing authors', dynamic_ncols=True):
        print(f"Processing author ID: {author_id}, Number of texts: {len(group)}")

        author_texts = group.drop(columns=['embedding_id', 'author'])

        if len(author_texts) > virtual_text_limit:
            sampled_indices = sample(range(len(author_texts)), virtual_text_limit)
            author_texts_sampled = author_texts.iloc[sampled_indices]
        else:
            author_texts_sampled = author_texts

        pairs, context_embeddings = generate_true_pairs_for_author(
            author_texts_sampled, author_id, context_index)

        for pair, embedding in zip(pairs[:max_pairs_limit], context_embeddings[:max_pairs_limit]):
            all_true_pairs.append([context_index, pair[1]])
            all_context_embeddings_with_index.append([context_index] + embedding)
            context_index += 1

        print(f"Total true pairs: {len(all_true_pairs)}")

    return all_true_pairs, all_context_embeddings_with_index

def generate_true_pairs_for_author(author_texts, author_id, start_context_index, max_context_size=3):
    true_pairs_list = []
    context_embeddings_list = []

    context_index = start_context_index

    for context_combination in itertools.combinations(author_texts.index, max_context_size):
        context_embedding = author_texts.loc[list(context_combination)].mean().values.tolist()
        context_embeddings_list.append(context_embedding + [author_id])

        remaining_texts = list(set(author_texts.index) - set(context_combination))
        for check_index in remaining_texts:
            check_embedding = author_texts.loc[check_index].values.tolist() + [author_id]
            true_pairs_list.append([context_index, check_embedding])
        context_index += 1

    return true_pairs_list, context_embeddings_list

datasetpath = "GG_70.csv"
df = pd.read_csv(datasetpath)
if df['author'].dtype == 'object':
    print("Author column is of type string")
else:
    print("Author column is of type numeric")


true_pairs_list, context_embeddings_with_index = generate_all_true_pairs(df)
true_pairs_df = pd.DataFrame(true_pairs_list, columns=['context_index', 'positive_embedding'])
context_embeddings_df = pd.DataFrame(context_embeddings_with_index, 
                                     columns=['context_index'] + [str(i) for i in range(len(context_embeddings_with_index[0]) - 2)] + ['author'])


true_pairs_df['positive_embedding'] = true_pairs_df['positive_embedding'].apply(lambda x: str(x[:-1]))
true_pairs_df = true_pairs_df.drop('context_index', axis=1)
context_embeddings_df['anchor_embedding'] = context_embeddings_df.iloc[:, 1:-1].apply(lambda row: str(row.tolist()), axis=1)
context_embeddings_df = context_embeddings_df.drop(['context_index', 'author'], axis=1)
combined_df = pd.concat([context_embeddings_df['anchor_embedding'], true_pairs_df], axis=1)


false_pairs_list = []
authors = df['author'].unique()

for index, context_row in tqdm(context_embeddings_df.iterrows(), total=context_embeddings_df.shape[0], desc='Generating false pairs', dynamic_ncols=True):
    anchor_embedding = context_row['anchor_embedding']
    anchor_author = context_embeddings_with_index[index][-1]
    
    other_authors = [author for author in authors if author != anchor_author]
    if not other_authors:
        print(f"Warning: No other authors found for context embedding at index {index}")
        continue
    
    negative_author = choice(other_authors)
    negative_texts = df[df['author'] == negative_author].drop(columns=['embedding_id', 'author'])
    negative_embed = negative_texts.sample(1).values.flatten().tolist()
    
    false_pairs_list.append({
        'anchor_embedding': anchor_embedding,
        'negative_embedding': str(negative_embed)
    })

false_pairs_df = pd.DataFrame(false_pairs_list)

triplets = pd.concat([combined_df, false_pairs_df['negative_embedding']], axis=1)

print("\nThe triplet has the following columns:", triplets.columns)
print(triplets.head())

triplets_shuffled = triplets.sample(frac=1).reset_index(drop=True)
triplets_shuffled.to_csv(r"Final-Triplets.csv", index=False)
print("Triplet dataset is saved!")









def evaluate(siamese_model, classifier_model, dataloader, triplet_criterion, bce_criterion, device):
    siamese_model.eval()
    classifier_model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    total_batches = len(dataloader)
    with torch.no_grad():
        for i, (anchor, positive, negative) in enumerate(dataloader, start=1):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            triplet_loss = triplet_criterion(anchor_out, positive_out, negative_out)
            
            classifier_out = classifier_model(anchor_out, positive_out, negative_out)
            bce_loss = bce_criterion(classifier_out, torch.ones_like(classifier_out))
            
            loss = triplet_loss + bce_loss
            running_loss += loss.item()
            
            predictions = (classifier_out > 0).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend([1] * len(predictions))
            
            print(f'Validation {i}/{total_batches}', end='\r')
    
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    
    if cm.shape[0] > 1:  
        label_0_accuracy = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0
        label_1_accuracy = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0
    else:  
        label_0_accuracy = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0
        label_1_accuracy = 0
    
    return running_loss / total_batches, overall_accuracy, label_0_accuracy, label_1_accuracy, precision, recall, f1, mcc
