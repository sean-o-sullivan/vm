import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

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

# Hyperparameters
input_size = 112
hidden_size = 256
lr = 0.001
batch_size = 128
num_epochs = 200

siamese_net = EnhancedSiameseNetwork(input_size, hidden_size).to(device)

# Use MarginRankingLoss instead of TripletMarginLoss
margin = 0.3
criterion = nn.MarginRankingLoss(margin=margin)
optimizer = optim.Adam(siamese_net.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

current_dir = os.getcwd()
train_dataset = TripletDataset(os.path.join(current_dir, "BnG_70.csv"))
val_dataset = TripletDataset(os.path.join(current_dir, "BnG_30.csv"))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

def train_epoch(siamese_model, dataloader, criterion, optimizer, device):
    siamese_model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
    for anchor, positive, negative in pbar:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        
        anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
        
        dist_pos = F.pairwise_distance(anchor_out, positive_out)
        dist_neg = F.pairwise_distance(anchor_out, negative_out)
        
        # Use 1 for positive pairs (should be ranked higher) and -1 for negative pairs
        target = torch.ones(anchor_out.size(0)).to(device)
        
        loss = criterion(dist_neg, dist_pos, target)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    return running_loss / len(dataloader)

def evaluate(siamese_model, dataloader, criterion, device):
    siamese_model.eval()
    running_loss = 0.0
    all_distances_pos = []
    all_distances_neg = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for anchor, positive, negative in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            
            dist_pos = F.pairwise_distance(anchor_out, positive_out)
            dist_neg = F.pairwise_distance(anchor_out, negative_out)
            
            target = torch.ones(anchor_out.size(0)).to(device)
            loss = criterion(dist_neg, dist_pos, target)
            
            running_loss += loss.item()
            
            all_distances_pos.extend(dist_pos.cpu().numpy())
            all_distances_neg.extend(dist_neg.cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    all_distances_pos = np.array(all_distances_pos)
    all_distances_neg = np.array(all_distances_neg)
    
    # Calculate distance-based metrics
    mean_pos_dist = np.mean(all_distances_pos)
    mean_neg_dist = np.mean(all_distances_neg)
    std_pos_dist = np.std(all_distances_pos)
    std_neg_dist = np.std(all_distances_neg)
    
    # Calculate overlap
    overlap_min = max(np.min(all_distances_pos), np.min(all_distances_neg))
    overlap_max = min(np.max(all_distances_pos), np.max(all_distances_neg))
    overlap_range = max(0, overlap_max - overlap_min)
    total_range = max(np.max(all_distances_pos), np.max(all_distances_neg)) - min(np.min(all_distances_pos), np.min(all_distances_neg))
    overlap_percentage = (overlap_range / total_range) * 100 if total_range > 0 else 0
    
    # Calculate AUC
    labels = np.concatenate([np.ones_like(all_distances_pos), np.zeros_like(all_distances_neg)])
    scores = -np.concatenate([all_distances_pos, all_distances_neg])  # Negative because smaller distance = more similar
    auc = roc_auc_score(labels, scores)
    
    return (running_loss / len(dataloader), mean_pos_dist, mean_neg_dist, std_pos_dist, std_neg_dist, 
            overlap_percentage, auc, all_distances_pos, all_distances_neg)

# Best model tracker
best_auc = float('-inf')
best_model_path = None

print("Starting Training!")
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}")
    
    train_loss = train_epoch(siamese_net, train_dataloader, criterion, optimizer, device)
    scheduler.step()
    
    val_loss, mean_pos_dist, mean_neg_dist, std_pos_dist, std_neg_dist, overlap_percentage, auc, all_distances_pos, all_distances_neg = evaluate(siamese_net, val_dataloader, criterion, device)
    
    print(f'Epoch {epoch+1}:')
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Mean Positive Distance: {mean_pos_dist:.4f} ± {std_pos_dist:.4f}')
    print(f'Mean Negative Distance: {mean_neg_dist:.4f} ± {std_neg_dist:.4f}')
    print(f'Distance Difference: {mean_neg_dist - mean_pos_dist:.4f}')
    print(f'Overlap Percentage: {overlap_percentage:.2f}%')
    print(f'AUC: {auc:.4f}')
    
    # Check if this is the best model so far based on AUC
    if auc > best_auc:
        best_auc = auc
        best_model_path = f"{current_dir}/best_distance_siamese_model.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': siamese_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'auc': auc,
            'overlap_percentage': overlap_percentage
        }, best_model_path)
        print(f"New best model found and saved at epoch {epoch+1} with AUC: {auc:.4f}")
    
    # saving every 10 epochs
    if (epoch + 1) % 10 == 0:
        model_save_path = f"{current_dir}/distance_siamese_model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': siamese_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, model_save_path)

print("Training completed!")
print(f"Best model saved at {best_model_path} with AUC: {best_auc:.4f}")
