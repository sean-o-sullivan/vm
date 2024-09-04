import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
import os
from tqdm import tqdm
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

def evaluate2(siamese_model, dataloader, criterion, device, threshold):
    siamese_model.eval()
    running_loss = 0.0
    all_positive_distances = []
    all_negative_distances = []
    positive_correct = 0
    negative_correct = 0
    total_positive = 0
    total_negative = 0
    
    flase_positive = 0
    flase_negative = 0
    pbar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for anchor, positive, negative in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            
            dist_pos = F.pairwise_distance(anchor_out, positive_out)
            dist_neg = F.pairwise_distance(anchor_out, negative_out)
            
            # Calculate triplet loss
            target = torch.ones(anchor_out.size(0)).to(device)
            loss = criterion(dist_neg, dist_pos, target)
            running_loss += loss.item()
            
            # Evaluation metrics
            all_positive_distances.extend(dist_pos.cpu().numpy())
            all_negative_distances.extend(dist_neg.cpu().numpy())
            
            positive_correct += torch.sum(dist_pos < threshold).item()
            negative_correct += torch.sum(dist_neg >= threshold).item()
            
            false_positive += torch.sum(dist_pos < threshold).item()
            false_negative += torch.sum(dist_neg >= threshold).item()
                        
            total_positive += len(dist_pos)
            total_negative += len(dist_neg)
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    avg_loss = running_loss / len(dataloader)
    positive_accuracy = positive_correct / total_positive if total_positive > 0 else 0
    negative_accuracy = negative_correct / total_negative if total_negative > 0 else 0
    overall_accuracy = (positive_correct + negative_correct) / (total_positive + total_negative) if (total_positive + total_negative) > 0 else 0
    
        # Compute precision, recall, and F1 score
    precision = positive_correct / (positive_correct + false_positive) if (positive_correct + false_positive) > 0 else 0
    recall = positive_correct / (positive_correct + false_negative) if (positive_correct + false_negative) > 0 else 0

    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Compute MCC (Matthews Correlation Coefficient)
    numerator = (positive_correct * negative_correct) - (false_positive * false_negative)
    denominator = ((positive_correct + false_positive) * (positive_correct + false_negative) *
                (negative_correct + false_positive) * (negative_correct + false_negative)) ** 0.5
    mcc = numerator / denominator if denominator > 0 else 0

    mean_pos_dist = np.mean(all_positive_distances)
    mean_neg_dist = np.mean(all_negative_distances)
    std_pos_dist = np.std(all_positive_distances)
    std_neg_dist = np.std(all_negative_distances)
    
    #  overlap
    overlap_min = max(np.min(all_positive_distances), np.min(all_negative_distances))
    overlap_max = min(np.max(all_positive_distances), np.max(all_negative_distances))
    overlap_range = max(0, overlap_max - overlap_min)
    total_range = max(np.max(all_positive_distances), np.max(all_negative_distances)) - min(np.min(all_positive_distances), np.min(all_negative_distances))
    overlap_percentage = (overlap_range / total_range) * 100 if total_range > 0 else 0
    
    #  AUC
    all_distances = np.concatenate([all_positive_distances, all_negative_distances])
    all_labels = np.concatenate([np.ones(len(all_positive_distances)), np.zeros(len(all_negative_distances))])
    auc = roc_auc_score(all_labels, -all_distances)  # Negative because smaller distance = more similar
    
    return avg_loss, mean_pos_dist, mean_neg_dist, std_pos_dist, std_neg_dist, overlap_percentage, auc, f1, precision,recall, mcc, overall_accuracy, positive_accuracy, negative_accuracy

def evaluate(siamese_model, dataloader, criterion, device, threshold):
    siamese_model.eval()
    running_loss = 0.0
    all_positive_distances = []
    all_negative_distances = []
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
   
    pbar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for anchor, positive, negative in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            dist_pos = F.pairwise_distance(anchor_out, positive_out)
            dist_neg = F.pairwise_distance(anchor_out, negative_out)
           
            # Calculate triplet loss
            target = torch.ones(anchor_out.size(0)).to(device)
            loss = criterion(dist_neg, dist_pos, target)
            running_loss += loss.item()
           
            # Evaluation metrics
            all_positive_distances.extend(dist_pos.cpu().numpy())
            all_negative_distances.extend(dist_neg.cpu().numpy())
           
            true_positive += torch.sum((dist_pos < threshold)).item()
            false_negative += torch.sum((dist_pos >= threshold)).item()
            true_negative += torch.sum((dist_neg >= threshold)).item()
            false_positive += torch.sum((dist_neg < threshold)).item()
           
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
   
    avg_loss = running_loss / len(dataloader)
    total_samples = true_positive + true_negative + false_positive + false_negative
   
    # Calculate accuracy metrics
    overall_accuracy = (true_positive + true_negative) / total_samples if total_samples > 0 else 0
    positive_accuracy = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    negative_accuracy = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
   
    # Compute precision, recall, and F1 score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
   
    # Compute MCC (Matthews Correlation Coefficient)
    numerator = (true_positive * true_negative) - (false_positive * false_negative)
    denominator = np.sqrt((true_positive + false_positive) * (true_positive + false_negative) *
                          (true_negative + false_positive) * (true_negative + false_negative))
    mcc = numerator / denominator if denominator > 0 else 0
   
    # Calculate distance statistics
    mean_pos_dist = np.mean(all_positive_distances)
    mean_neg_dist = np.mean(all_negative_distances)
    std_pos_dist = np.std(all_positive_distances)
    std_neg_dist = np.std(all_negative_distances)
   
    # Calculate overlap
    overlap_min = max(np.min(all_positive_distances), np.min(all_negative_distances))
    overlap_max = min(np.max(all_positive_distances), np.max(all_negative_distances))
    overlap_range = max(0, overlap_max - overlap_min)
    total_range = max(np.max(all_positive_distances), np.max(all_negative_distances)) - min(np.min(all_positive_distances), np.min(all_negative_distances))
    overlap_percentage = (overlap_range / total_range) * 100 if total_range > 0 else 0
   
    # Calculate AUC
    all_distances = np.concatenate([all_positive_distances, all_negative_distances])
    all_labels = np.concatenate([np.ones(len(all_positive_distances)), np.zeros(len(all_negative_distances))])
    auc = roc_auc_score(all_labels, -all_distances)  # Negative because smaller distance = more similar
   
    return {
        'avg_loss': avg_loss,
        'mean_pos_dist': mean_pos_dist,
        'mean_neg_dist': mean_neg_dist,
        'std_pos_dist': std_pos_dist,
        'std_neg_dist': std_neg_dist,
        'overlap_percentage': overlap_percentage,
        'auc': auc,
        'f1': f1_score,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'overall_accuracy': overall_accuracy,
        'positive_accuracy': positive_accuracy,
        'negative_accuracy': negative_accuracy
    }







# Hyperparameters
input_size = 112
hidden_size = 256
batch_size = 128

# model
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "BnG_10_best_transformer_siamese_model.pth")
checkpoint = torch.load(model_path, map_location=device)

siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)
siamese_net.load_state_dict(checkpoint['model_state_dict'])
siamese_net.eval()

threshold = checkpoint['threshold']

#val_set = "BnG_2_30.csv"


#val_set = "Final-Triplets_B_30_|3|_VTL5_C3.csv"
val_set = "Final-Triplets_G_30_|3|_VTL5_C4.csv"


# Load the BnG_30 dataset
val_dataset = TripletDataset(os.path.join(current_dir, val_set))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

# Use MarginRankingLoss (same as in training)
margin = 0.05
criterion = nn.MarginRankingLoss(margin=margin)

print("Starting Evaluation...")

# Evaluate the model
evaluation_results = evaluate(siamese_net, val_dataloader, criterion, device, threshold)

print(f'Evaluation Results:')
print(f'Loss: {evaluation_results["avg_loss"]}')
print(f'Mean Positive Distance: {evaluation_results["mean_pos_dist"]} ± {evaluation_results["std_pos_dist"]}')
print(f'Mean Negative Distance: {evaluation_results["mean_neg_dist"]} ± {evaluation_results["std_neg_dist"]}')
print(f'Overlap Percentage: {evaluation_results["overlap_percentage"]}%')
print(f'AUC: {evaluation_results["auc"]}')
print(f'F1: {evaluation_results["f1"]}')
print(f'Precision: {evaluation_results["precision"]}')
print(f'Recall: {evaluation_results["recall"]}')
print(f'MCC: {evaluation_results["mcc"]}')
print(f'Overall Accuracy: {evaluation_results["overall_accuracy"]} (Threshold: {threshold})')
print(f'Positive Accuracy: {evaluation_results["positive_accuracy"]}, Negative Accuracy: {evaluation_results["negative_accuracy"]}')

# Save results to a file
results_file = f"BnG_30_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(results_file, 'w') as f:
    f.write(f'Evaluation Results:\n')
    f.write(f'Loss: {evaluation_results["avg_loss"]}\n')
    f.write(f'Mean Positive Distance: {evaluation_results["mean_pos_dist"]} ± {evaluation_results["std_pos_dist"]}\n')
    f.write(f'Mean Negative Distance: {evaluation_results["mean_neg_dist"]} ± {evaluation_results["std_neg_dist"]}\n')
    f.write(f'Distance Difference: {evaluation_results["mean_neg_dist"] - evaluation_results["mean_pos_dist"]}\n')
    f.write(f'Overlap Percentage: {evaluation_results["overlap_percentage"]:.2f}%\n')
    f.write(f'AUC: {evaluation_results["auc"]}\n')
    f.write(f'F1: {evaluation_results["f1"]}\n')
    f.write(f'Precision: {evaluation_results["precision"]}\n')
    f.write(f'Recall: {evaluation_results["recall"]}\n')
    f.write(f'MCC: {evaluation_results["mcc"]}\n')
    f.write(f'Overall Accuracy: {evaluation_results["overall_accuracy"]} (Threshold: {threshold})\n')
    f.write(f'Positive Accuracy: {evaluation_results["positive_accuracy"]}, Negative Accuracy: {evaluation_results["negative_accuracy"]}\n')

print(f"\nEvaluation completed! Results saved to {results_file}")