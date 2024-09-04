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
    def __init__(self, input_size, hidden_size, num_layers=2, num_heads=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.norm = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(0)  # add sequence dimensiona
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # remove sequence dimension
        x = self.output_proj(x)
        x = self.norm(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization

class SiameseTransformerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseTransformerNetwork, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size)

    def forward(self, anchor, positive, negative):
        anchor_out = self.encoder(anchor)
        positive_out = self.encoder(positive)
        negative_out = self.encoder(negative)
        return anchor_out, positive_out, negative_out
    

class EvaluationDataset(Dataset):
    def __init__(self, csv_file, column):
        self.data = pd.read_csv(csv_file)
        self.column = column
        self.valid_indices = self._get_valid_indices()
        print(f"Processing column: {column}")
        print(f"Total samples: {len(self.data)}, Valid samples: {len(self.valid_indices)}")
        print(f"Skipped {len(self.data) - len(self.valid_indices)} samples due to invalid comparison embeddings.")

    def _get_valid_indices(self):
        return [i for i, row in self.data.iterrows()
                if self._is_valid_embedding(row[self.column])]

    def _is_valid_embedding(self, embedding_str):
        try:
            embedding = ast.literal_eval(embedding_str)
            return embedding != [1] and len(embedding) == 112
        except (ValueError, SyntaxError):
            return False

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.data.iloc[self.valid_indices[idx]]
        anchor_embedding = self._parse_embedding(row['anchor_embedding'])
        comparison_embedding = self._parse_embedding(row[self.column])
        return (torch.tensor(anchor_embedding, dtype=torch.float32),
                torch.tensor(comparison_embedding, dtype=torch.float32))

    def _parse_embedding(self, embedding_str):
        return ast.literal_eval(embedding_str)
    

def evaluate_model(model, dataloader, device, threshold=0.99):
    model.eval()
    all_distances = []
    all_predictions = []
    
    with torch.no_grad():
        for anchor, comparison in tqdm(dataloader, desc="Evaluating"):
            anchor, comparison = anchor.to(device), comparison.to(device)
            
            dummy = torch.zeros_like(anchor).to(device)
            anchor_out, comparison_out, _ = model(anchor, comparison, dummy)
            
            dist = F.pairwise_distance(anchor_out, comparison_out)
            
            all_distances.extend(dist.cpu().numpy())
            all_predictions.extend((dist >= threshold).cpu().numpy().astype(int))
    
    return all_distances, all_predictions

input_size = 112
hidden_size = 256
batch_size = 1 #128, we are doing evaluation, even though it should technically be fine for both, and it is.
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "BnG_10_best_transformer_siamese_model.pth")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)
siamese_net.load_state_dict(checkpoint['model_state_dict'])
siamese_net.eval()

# Define the columns to evaluate
embedding_columns = [
    'mimic_GPT3AGG_embedding', 'mimic_GPT4TAGG_embedding',
    'mimic_GPT4oAGG_embedding', 'topic_GPT3AGG_embedding',
    'topic_GPT4TAGG_embedding', 'topic_GPT4oAGG_embedding'
]

print("Starting Evaluation...")

# Prepare CSV file for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"dissimilarity_evaluation_results_{timestamp}.csv"
with open(results_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Embedding Type', 'Accuracy', 'Precision', 'Recall', 'F1 Score',
                        'Mean Distance', 'Std Distance', 'Min Distance', 'Max Distance',
                        'Threshold', 'Total Samples',
                        'True Negatives', 'False Positives',
                        'True Positive Rate', 'False Positive Rate'])

    for column in embedding_columns:
        print(f"\nEvaluating {column}:")
        eval_dataset = EvaluationDataset('/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/GPT/output_S.csv', column)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4)
        distances, predictions = evaluate_model(siamese_net, eval_dataloader, device, threshold=checkpoint['threshold'])#checkpoint['threshold']
        
        # Calculate metrics
        total_samples = len(predictions)
        true_negatives = sum(predictions)
        false_positives = total_samples - true_negatives  
        
        accuracy = accuracy_score([1] * total_samples, predictions)
        precision = precision_score([1] * total_samples, predictions, zero_division=1)
        recall = recall_score([1] * total_samples, predictions, zero_division=1)
        f1 = f1_score([1] * total_samples, predictions, zero_division=1)


        true_positive_rate = 0  # We don't expect any true positives
        false_positive_rate = false_positives / total_samples
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
    
        # csvwriter.writerow([column, accuracy, precision, recall, f1,
        #                     mean_dist, std_dist, min_dist, max_dist,
        #                     checkpoint['threshold'], total_samples,
        #                     true_negatives, false_positives,
        #                     true_positive_rate, false_positive_rate])
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Mean Distance: {mean_dist:.4f} ± {std_dist:.4f}")
        print(f"Min Distance: {min_dist:.4f}")
        print(f"Max Distance: {max_dist:.4f}")
        print(f"Threshold: {checkpoint['threshold']:.4f}")
        print(f"Total Samples: {total_samples}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Positives: {false_positives}")
        print(f"True Positive Rate: {true_positive_rate:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")

print(f"\nDissimilarity evaluation completed! Results saved to {results_file}")
