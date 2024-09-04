import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

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
        x = self.input_proj(x).unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        x = self.output_proj(x)
        x = self.norm(x)
        return F.normalize(x, p=2, dim=1)

class SiameseTransformerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseTransformerNetwork, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size)

    def forward(self, anchor, positive, negative):
        anchor_out = self.encoder(anchor)
        positive_out = self.encoder(positive)
        negative_out = self.encoder(negative)
        return anchor_out, positive_out, negative_out

class TripletDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        anchor_embedding = ast.literal_eval(row['anchor_embedding'])
        positive_embedding = ast.literal_eval(row['positive_embedding'])
        negative_embedding = ast.literal_eval(row['negative_embedding'])
        return (torch.tensor(anchor_embedding, dtype=torch.float32),
                torch.tensor(positive_embedding, dtype=torch.float32),
                torch.tensor(negative_embedding, dtype=torch.float32))

def compute_distances(siamese_model, dataloader, device):
    siamese_model.eval()
    all_positive_distances = []
    all_negative_distances = []
    
    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc="Computing distances"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            
            dist_pos = F.pairwise_distance(anchor_out, positive_out)
            dist_neg = F.pairwise_distance(anchor_out, negative_out)
            
            all_positive_distances.extend(dist_pos.cpu().numpy())
            all_negative_distances.extend(dist_neg.cpu().numpy())
    
    return all_positive_distances, all_negative_distances

def plot_roc_curve(fpr, tpr, roc_auc, threshold):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Add threshold point
    threshold_index = np.argmin(np.abs(fpr + tpr - 1))
    plt.plot(fpr[threshold_index], tpr[threshold_index], 'ro', label=f'Threshold = {threshold:.4f}')
    plt.legend(loc="lower right")
    
    plt.savefig('roc_curve.png')
    print("ROC curve saved as 'roc_curve.png'")

def evaluate(siamese_model, dataloader, device, threshold):
    siamese_model.eval()
    all_positive_distances = []
    all_negative_distances = []
    
    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc="Evaluating"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            
            dist_pos = F.pairwise_distance(anchor_out, positive_out)
            dist_neg = F.pairwise_distance(anchor_out, negative_out)
            
            all_positive_distances.extend(dist_pos.cpu().numpy())
            all_negative_distances.extend(dist_neg.cpu().numpy())
    
    all_distances = np.concatenate([all_positive_distances, all_negative_distances])
    all_labels = np.concatenate([np.ones(len(all_positive_distances)), np.zeros(len(all_negative_distances))])
    
    predictions = (all_distances >= threshold).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    
    mean_pos_dist = np.mean(all_positive_distances)
    mean_neg_dist = np.mean(all_negative_distances)
    std_pos_dist = np.std(all_positive_distances)
    std_neg_dist = np.std(all_negative_distances)
    
    return accuracy, mean_pos_dist, mean_neg_dist, std_pos_dist, std_neg_dist

input_size = 112
hidden_size = 256
batch_size = 128

# model
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "BnG_2_best_transformer_siamese_model.pth")
checkpoint = torch.load(model_path, map_location=device)

siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)
siamese_net.load_state_dict(checkpoint['model_state_dict'])
siamese_net.eval()

val_dataset = TripletDataset(os.path.join(current_dir, "BnG_30.csv"))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

print("Computing distances for AUROC...")
all_positive_distances, all_negative_distances = compute_distances(siamese_net, val_dataloader, device)

all_distances = np.concatenate([all_positive_distances, all_negative_distances])
all_labels = np.concatenate([np.ones(len(all_positive_distances)), np.zeros(len(all_negative_distances))])

fpr, tpr, thresholds = roc_curve(all_labels, -all_distances)  # Negative distances because smaller distance = more similar
roc_auc = auc(fpr, tpr)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.4f}")

# Plot ROC curve
plot_roc_curve(fpr, tpr, roc_auc, optimal_threshold)

print("Evaluating with optimal threshold...")
accuracy, mean_pos_dist, mean_neg_dist, std_pos_dist, std_neg_dist = evaluate(siamese_net, val_dataloader, device, optimal_threshold)

print(f'Evaluation Results:')
print(f'AUC: {roc_auc:.4f}')
print(f'Optimal Threshold: {optimal_threshold:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Mean Positive Distance: {mean_pos_dist:.4f} ± {std_pos_dist:.4f}')
print(f'Mean Negative Distance: {mean_neg_dist:.4f} ± {std_neg_dist:.4f}')
print(f'Distance Difference: {mean_neg_dist - mean_pos_dist:.4f}')

results_file = f"BnG_30_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(results_file, 'w') as f:
    f.write(f'Evaluation Results:\n')
    f.write(f'AUC: {roc_auc:.4f}\n')
    f.write(f'Optimal Threshold: {optimal_threshold:.4f}\n')
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(f'Mean Positive Distance: {mean_pos_dist:.4f} ± {std_pos_dist:.4f}\n')
    f.write(f'Mean Negative Distance: {mean_neg_dist:.4f} ± {std_neg_dist:.4f}\n')
    f.write(f'Distance Difference: {mean_neg_dist - mean_pos_dist:.4f}\n')

print(f"\nEvaluation is finally completed! Results saved to {results_file}")
