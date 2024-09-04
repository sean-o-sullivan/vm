import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os
from tqdm import tqdm
import csv
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnhancedEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EnhancedEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.bn4 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.bn4(self.fc4(x))
        return F.normalize(x, p=2, dim=1)  # L2 normalization

class EnhancedSiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EnhancedSiameseNetwork, self).__init__()
        self.encoder = EnhancedEncoder(input_size, hidden_size)

    def forward(self, anchor, negative):
        anchor_out = self.encoder(anchor)
        negative_out = self.encoder(negative)
        return anchor_out, negative_out

class EvaluationDataset(Dataset):
    def __init__(self, csv_file, column):
        self.data = pd.read_csv(csv_file)
        self.column = column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        anchor_embedding = ast.literal_eval(row['anchor_embedding'])
        negative_embedding = ast.literal_eval(row[self.column])
        
        return (torch.tensor(anchor_embedding, dtype=torch.float32),
                torch.tensor(negative_embedding, dtype=torch.float32))

def evaluate_model(model, dataloader, device, threshold=1.5):
    model.eval()
    all_distances = []
    all_predictions = []
    
    with torch.no_grad():
        for anchor, negative in tqdm(dataloader, desc="Evaluating"):
            anchor, negative = anchor.to(device), negative.to(device)
            
            anchor_out, negative_out = model(anchor, negative)
            
            dist = F.pairwise_distance(anchor_out, negative_out)
            
            all_distances.extend(dist.cpu().numpy())
            all_predictions.extend((dist >= threshold).cpu().numpy().astype(int))
    
    return all_distances, all_predictions

# Hyperparameters
input_size = 112
hidden_size = 256
batch_size = 128

# Load the model
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "BnG_2_best_distance_siamese_model.pth")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

siamese_net = EnhancedSiameseNetwork(input_size, hidden_size).to(device)
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
results_file = f"c2omprehensive_evaluation_results_{timestamp}.csv"
with open(results_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Embedding Type', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score',
                        'Mean Distance', 'Std Distance', 'Min Distance', 'Max Distance',
                        'Threshold', 'Unique Predictions', 'Total Samples',
                        'Positive Predictions', 'Negative Predictions',
                        'True Positives', 'False Positives', 'True Negatives', 'False Negatives'])

    for column in embedding_columns:
        print(f"\nEvaluating {column}:")
        
        # Load the evaluation dataset for this column
        eval_dataset = EvaluationDataset('/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/GPT/output_S.csv', column)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4)
        
        # Evaluate the model
        distances, predictions = evaluate_model(siamese_net, eval_dataloader, device, threshold=1.2)#checkpoint['threshold']
        
        # Calculate metrics
        true_labels = [1] * len(predictions)  # All samples are supposed to be negative class
        accuracy = accuracy_score(true_labels, predictions)
        unique_predictions = len(set(predictions))
        total_samples = len(predictions)
        positive_predictions = sum(predictions)
        negative_predictions = total_samples - positive_predictions
        
        # Confusion matrix elements
        true_positives = sum((t == 1 and p == 1) for t, p in zip(true_labels, predictions))
        false_positives = sum((t == 0 and p == 1) for t, p in zip(true_labels, predictions))
        true_negatives = sum((t == 0 and p == 0) for t, p in zip(true_labels, predictions))
        false_negatives = sum((t == 1 and p == 0) for t, p in zip(true_labels, predictions))
        
        # Handle metric calculations
    #   # if unique_predictions > 1:
    #     if 2 > 1:
    #         auc = roc_auc_score(true_labels, distances)
    #         precision = precision_score(true_labels, predictions)
    #         recall = recall_score(true_labels, predictions)
    #         f1 = f1_score(true_labels, predictions)
    #     else:
        auc = precision = recall = f1 = "N/A (All predictions are the same class)"
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # # Write results to CSV
        # csvwriter.writerow([column, accuracy, auc, precision, recall, f1,
        #                     mean_dist, std_dist, min_dist, max_dist,
        #                     checkpoint['threshold'], unique_predictions, total_samples,
        #                     positive_predictions, negative_predictions,
        #                     true_positives, false_positives, true_negatives, false_negatives])
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Mean Distance: {mean_dist:.4f} ± {std_dist:.4f}")
        print(f"Min Distance: {min_dist:.4f}")
        print(f"Max Distance: {max_dist:.4f}")
        print(f"Threshold: {checkpoint['threshold']:.4f}")
        print(f"Unique Predictions: {unique_predictions}")
        print(f"Total Samples: {total_samples}")
        print(f"Positive Predictions: {positive_predictions}")
        print(f"Negative Predictions: {negative_predictions}")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Negatives: {false_negatives}")

print(f"\nComprehensive evaluation completed! Results saved to {results_file}")
