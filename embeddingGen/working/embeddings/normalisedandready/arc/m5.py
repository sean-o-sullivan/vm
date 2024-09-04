import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

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

triplet_criterion = nn.TripletMarginLoss(margin=0.3)
optimizer = optim.Adam(siamese_net.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

current_dir = os.getcwd()
train_dataset = TripletDataset(os.path.join(current_dir, "Final-Triplets_G_70_|_VTL5_C3.csv"))
val_dataset = TripletDataset(os.path.join(current_dir, "Final-Triplets_G_30_|_VTL5_C3.csv"))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

def train_epoch(siamese_model, dataloader, triplet_criterion, optimizer, device):
    siamese_model.train()
    running_loss = 0.0
    for anchor, positive, negative in dataloader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        
        anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
        loss = triplet_criterion(anchor_out, positive_out, negative_out)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(siamese_model, dataloader, triplet_criterion, device):
    siamese_model.eval()
    running_loss = 0.0
    all_distances_pos = []
    all_distances_neg = []
    
    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            loss = triplet_criterion(anchor_out, positive_out, negative_out)
            running_loss += loss.item()
            
            # Calculate distances
            dist_pos = F.pairwise_distance(anchor_out, positive_out)
            dist_neg = F.pairwise_distance(anchor_out, negative_out)
            
            all_distances_pos.extend(dist_pos.cpu().numpy())
            all_distances_neg.extend(dist_neg.cpu().numpy())
    
    all_distances_pos = np.array(all_distances_pos)
    all_distances_neg = np.array(all_distances_neg)
    
    # Calculate separate thresholds for positive and negative pairs
    threshold_pos = np.percentile(all_distances_pos, 95)  # 95th percentile for positive pairs
    threshold_neg = np.percentile(all_distances_neg, 5)   # 5th percentile for negative pairs
    
    # Use the average of the two thresholds
    best_threshold = (threshold_pos + threshold_neg) / 2
    
    # Combine distances and create labels
    all_distances = np.concatenate([all_distances_pos, all_distances_neg])
    all_labels = np.concatenate([np.ones_like(all_distances_pos), np.zeros_like(all_distances_neg)])
    
    # Make predictions based on the best threshold
    predictions = (all_distances < best_threshold).astype(int)
    
    # Calculate evaluation metrics
    try:
        accuracy = accuracy_score(all_labels, predictions)
        precision = precision_score(all_labels, predictions)
        recall = recall_score(all_labels, predictions)
        f1 = f1_score(all_labels, predictions)
        mcc = matthews_corrcoef(all_labels, predictions)
    except Exception as e:
        print(f"Error in calculating metrics: {e}")
        return running_loss / len(dataloader), 0, 0, 0, 0, 0
    
    return running_loss / len(dataloader), accuracy, precision, recall, f1, mcc



print("Starting Training!")
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}")
    train_loss = train_epoch(siamese_net, train_dataloader, triplet_criterion, optimizer, device)
    scheduler.step()
    
    val_loss, accuracy, precision, recall, f1, mcc = evaluate(siamese_net, val_dataloader, triplet_criterion, device)
    
    print(f'Epoch {epoch+1}:')
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}')

    if (epoch + 1) % 10 == 0:
        model_save_path = f"{current_dir}/enhanced_siamese_model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': siamese_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, model_save_path)

print("Training completed!")
