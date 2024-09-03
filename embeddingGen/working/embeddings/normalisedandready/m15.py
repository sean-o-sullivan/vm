import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from tqdm import tqdm
import csv
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)  #sequence dimension
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # sequence dimension
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization

class TransformerSiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TransformerSiameseNetwork, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size)

    def forward(self, anchor, comparison):
        anchor_out = self.encoder(anchor)
        comparison_out = self.encoder(comparison)
        return anchor_out, comparison_out

class EvaluationDataset(Dataset):
    def __init__(self, csv_file, column):
        self.data = pd.read_csv(csv_file)
        self.column = column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        anchor_embedding = ast.literal_eval(row['anchor_embedding'])
        comparison_embedding = ast.literal_eval(row[self.column])
        
        return (torch.tensor(anchor_embedding, dtype=torch.float32),
                torch.tensor(comparison_embedding, dtype=torch.float32))

def evaluate_model(model, dataloader, device, threshold=0.99):
    model.eval()
    all_distances = []
    all_predictions = []
    
    with torch.no_grad():
        for anchor, comparison in tqdm(dataloader, desc="Evaluating"):
            anchor, comparison = anchor.to(device), comparison.to(device)
            
            anchor_out, comparison_out = model(anchor, comparison)
            
            dist = F.pairwise_distance(anchor_out, comparison_out)
            
            all_distances.extend(dist.cpu().numpy())
            all_predictions.extend((dist >= threshold).cpu().numpy().astype(int))
    
    return all_distances, all_predictions

input_size = 112
hidden_size = 256
batch_size = 128
siamese_net = TransformerSiameseNetwork(input_size, hidden_size).to(device)

# to evaluate
embedding_columns = [
    'mimic_GPT3AGG_embedding', 'mimic_GPT4TAGG_embedding',
    'mimic_GPT4oAGG_embedding', 'topic_GPT3AGG_embedding',
    'topic_GPT4TAGG_embedding', 'topic_GPT4oAGG_embedding'
]

print("Starting Evaluation...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"transformer_dissimilarity_evaluation_results_{timestamp}.csv"
with open(results_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Embedding Type', 'Accuracy', 'Precision', 'Recall', 'F1 Score',
                        'Mean Distance', 'Std Distance', 'Min Distance', 'Max Distance',
                        'Threshold', 'Total Samples',
                        'True Negatives', 'False Positives',
                        'True Positive Rate', 'False Positive Rate'])

    for column in embedding_columns:
        print(f"\nEvaluating {column}:")
        
        eval_dataset = EvaluationDataset('/path/to/your/csv/file.csv', column)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4)
        distances, predictions = evaluate_model(siamese_net, eval_dataloader, device, threshold=0.5)
        
        total_samples = len(predictions)
        true_negatives = sum(predictions)  #
        false_positives = total_samples - true_negatives 
        
        accuracy = accuracy_score([1] * total_samples, predictions)
        precision = precision_score([1] * total_samples, predictions, zero_division=1)
        recall = recall_score([1] * total_samples, predictions, zero_division=1)
        f1 = f1_score([1] * total_samples, predictions, zero_division=1)
        
        true_positive_rate = 0  
        false_positive_rate = false_positives / total_samples
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        csvwriter.writerow([column, accuracy, precision, recall, f1,
                            mean_dist, std_dist, min_dist, max_dist,
                            0.5, total_samples,
                            true_negatives, false_positives,
                            true_positive_rate, false_positive_rate])

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Mean Distance: {mean_dist:.4f} ± {std_dist:.4f}")
        print(f"Min Distance: {min_dist:.4f}")
        print(f"Max Distance: {max_dist:.4f}")
        print(f"Threshold: 0.5")
        print(f"Total Samples: {total_samples}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Positives: {false_positives}")
        print(f"True Positive Rate: {true_positive_rate:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")

print(f"\nTransformer-based dissimilarity evaluation arcompleted! Results saved to {results_file}")
