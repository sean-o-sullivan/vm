import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import os
from tqdm import tqdm

# Define the model architecture (same as in the training script)
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

def evaluate(siamese_model, dataloader, triplet_criterion, device):
    siamese_model.eval()
    running_loss = 0.0
    all_distances_pos = []
    all_distances_neg = []
    all_triplet_losses = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for anchor, positive, negative in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            loss = triplet_criterion(anchor_out, positive_out, negative_out)
            running_loss += loss.item()
            
            dist_pos = F.pairwise_distance(anchor_out, positive_out)
            dist_neg = F.pairwise_distance(anchor_out, negative_out)
            
            all_distances_pos.extend(dist_pos.cpu().numpy())
            all_distances_neg.extend(dist_neg.cpu().numpy())
            
            # Calculate individual triplet losses
            triplet_losses = F.relu(dist_pos - dist_neg + triplet_criterion.margin)
            all_triplet_losses.extend(triplet_losses.cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    all_distances_pos = np.array(all_distances_pos)
    all_distances_neg = np.array(all_distances_neg)
    all_triplet_losses = np.array(all_triplet_losses)
    
    # Calculate the fraction of triplets where positive distance is smaller than negative
    correct_order = np.sum(all_distances_pos < all_distances_neg) / len(all_distances_pos)
    
    # Calculate the fraction of triplets with zero loss
    zero_loss_fraction = np.sum(all_triplet_losses == 0) / len(all_triplet_losses)
    
    # Use correct ordering as our primary accuracy metric
    accuracy = correct_order
    
    # For other metrics, we'll use a threshold-based approach
    all_distances = np.concatenate([all_distances_pos, all_distances_neg])
    all_labels = np.concatenate([np.ones_like(all_distances_pos), np.zeros_like(all_distances_neg)])
    
    best_threshold = np.mean(all_distances)
    predictions = (all_distances < best_threshold).astype(int)
    
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    mcc = matthews_corrcoef(all_labels, predictions)
    
    return (running_loss / len(dataloader), accuracy, precision, recall, f1, mcc, 
            all_distances_pos, all_distances_neg, zero_loss_fraction)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (same as in training)
input_size = 112
hidden_size = 256
batch_size = 128

# Initialize model
siamese_net = EnhancedSiameseNetwork(input_size, hidden_size).to(device)

# Load the saved model
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "enhanced_siamese_model_epoch_150.pth")
checkpoint = torch.load(model_path, map_location=device)
siamese_net.load_state_dict(checkpoint['model_state_dict'])

# Load validation dataset
val_dataset = TripletDataset(os.path.join(current_dir, "BnG_30.csv"))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

# Set up loss function
triplet_criterion = nn.TripletMarginLoss(margin=0.3)

# Evaluate the model
val_loss, accuracy, precision, recall, f1, mcc, all_distances_pos, all_distances_neg, zero_loss_fraction = evaluate(siamese_net, val_dataloader, triplet_criterion, device)

print("Evaluation Results:")
print(f'Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}')
print(f'Zero Loss Fraction: {zero_loss_fraction:.4f}')
print(f"Positive distances: min={np.min(all_distances_pos):.4f}, max={np.max(all_distances_pos):.4f}, mean={np.mean(all_distances_pos):.4f}")
print(f"Negative distances: min={np.min(all_distances_neg):.4f}, max={np.max(all_distances_neg):.4f}, mean={np.mean(all_distances_neg):.4f}")

# Additional analysis to check for overfitting
print("\nAdditional Analysis:")
print(f"Positive distances std: {np.std(all_distances_pos):.4f}")
print(f"Negative distances std: {np.std(all_distances_neg):.4f}")
print(f"Overlap fraction: {np.sum((all_distances_pos > np.min(all_distances_neg)) & (all_distances_pos < np.max(all_distances_neg))) / len(all_distances_pos):.4f}")
