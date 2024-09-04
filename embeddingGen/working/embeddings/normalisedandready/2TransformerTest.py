import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, matthews_corrcoef
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
        x = self.input_proj(x).unsqueeze(0)  # add sequence dimension
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # remove sequence dimension
        x = self.output_proj(x)
        x = self.norm(x)
        return F.normalize(x, p=2, dim=1)  # L2

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import matthews_corrcoef
import os
from tqdm import tqdm
import csv
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define accuracy on the positive (human) class at the beginning, this is for calculating the other metrics..
POSITIVE_ACCURACY = 0.88688
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

def calculate_custom_metrics(y_pred, total_samples, positive_accuracy):
    true_negatives = sum(y_pred)
    false_positives = total_samples - true_negatives
   
    # Estimate true positives and false negatives based on positive_accuracy
    estimated_positives = total_samples  # Assuming balanced dataset
    true_positives = int(estimated_positives * positive_accuracy)
    false_negatives = estimated_positives - true_positives
   
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / (2 * total_samples)
    neg_acc = true_negatives /(true_negatives+false_positives)
    pos_acc = true_positives /(true_positives+false_negatives)
    # Calculate MCC
    mcc_numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
    mcc_denominator = np.sqrt((true_positives + false_positives) * (true_positives + false_negatives) *
                              (true_negatives + false_positives) * (true_negatives + false_negatives))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0
   
    # Calculate AUC using the trapezoid rule
    fpr = false_positives / (false_positives + true_negatives)
    tpr = true_positives / (true_positives + false_negatives)
    auc = 0.5 * (1 + tpr - fpr)
   
    return accuracy, pos_acc,neg_acc, precision, recall, f1, mcc, auc, true_positives, false_negatives, true_negatives, false_positives, tpr, fpr

# Hyperparameters
input_size = 112
hidden_size = 256
batch_size = 1 #128

# Load the model
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "BnG_10_best_transformer_siamese_model.pth")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)
siamese_net.load_state_dict(checkpoint['model_state_dict'])
siamese_net.eval()

# Define the columns to evaluate
embedding_columns = [
    'mimic_GPT3ABB_embedding', 'mimic_GPT4TABB_embedding',
    'mimic_GPT4oABB_embedding', 'topic_GPT3ABB_embedding',
    'topic_GPT4TABB_embedding', 'topic_GPT4oABB_embedding'
]

print("Starting Evaluation...")

# Prepare CSV file for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"dissimilarity_evaluation_results_{timestamp}.csv"
with open(results_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Embedding Type', 'Accuracy', "pos_acc", "neg_acc",'Precision', 'Recall', 'F1 Score', 'MCC', 'AUC',
                        'Mean Distance', 'Std Distance', 'Min Distance', 'Max Distance',
                        'Threshold', 'Total Samples',
                        'True Positives', 'False Negatives', 'True Negatives', 'False Positives',
                        'True Positive Rate', 'False Positive Rate'])

    for column in embedding_columns:
        print(f"\nEvaluating {column}:")
       
        # Load 
        eval_dataset = EvaluationDataset('/home/aiadmin/Desktop/code/vm/embeddingGen/working/embeddings/normalisedandready/GPT/C1_BB_output_S.csv', column)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4)
       
        # Evaluate
        distances, predictions = evaluate_model(siamese_net, eval_dataloader, device, threshold=checkpoint['threshold'])
       
        # Calculate
        total_samples = len(predictions)
        accuracy, pos_acc, neg_acc, precision, recall, f1, mcc, auc, true_positives, false_negatives, true_negatives, false_positives, tpr, fpr = calculate_custom_metrics(predictions, total_samples, POSITIVE_ACCURACY)
       
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
       
        # Write 
        csvwriter.writerow([column, accuracy, pos_acc, neg_acc, precision, recall, f1, mcc, auc,
                            mean_dist, std_dist, min_dist, max_dist,
                            checkpoint['threshold'], total_samples,
                            true_positives, false_negatives, true_negatives, false_positives,
                            tpr, fpr])
       
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Positive Class accuracy: {pos_acc:.4f}")
        print(f"Negative Class accuracy: {neg_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Mean Distance: {mean_dist:.4f} ± {std_dist:.4f}")
        print(f"Min Distance: {min_dist:.4f}")
        print(f"Max Distance: {max_dist:.4f}")
        print(f"Threshold: {checkpoint['threshold']:.4f}")
        print(f"Total Samples: {total_samples}")
        print(f"True Positives: {true_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Positives: {false_positives}")
        print(f"True Positive Rate: {tpr:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")

print(f"\nDissimilarity evaluation completed! Results saved to {results_file}")