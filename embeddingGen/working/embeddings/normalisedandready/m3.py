import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
import os
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Improved Encoder
class ImprovedEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImprovedEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(self.fc3(x))
        return x

# Improved Network
class ImprovedSiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImprovedSiameseNetwork, self).__init__()
        self.encoder = ImprovedEncoder(input_size, hidden_size)

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

input_size = 112
hidden_size = 128
lr = 0.0001
batch_size = 64
num_epochs = 100

# Initialize
siamese_net = ImprovedSiameseNetwork(input_size, hidden_size).to(device)

# Loss and optimizer
triplet_criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
optimizer = optim.Adam(siamese_net.parameters(), lr=lr, weight_decay=1e-5)

# data
current_dir = os.getcwd()
train_dataset = TripletDataset(os.path.join(current_dir, "Final-Triplets_G_70_|_VTL5_C3.csv"))
val_dataset = TripletDataset(os.path.join(current_dir, "Final-Triplets_G_30_|_VTL5_C3.csv"))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

# Training 
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

# Evaluation 
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
            
            dist_pos = F.pairwise_distance(anchor_out, positive_out)
            dist_neg = F.pairwise_distance(anchor_out, negative_out)
            
            all_distances_pos.extend(dist_pos.cpu().numpy())
            all_distances_neg.extend(dist_neg.cpu().numpy())
    
    all_distances = np.concatenate([all_distances_pos, all_distances_neg])
    all_labels = np.concatenate([np.ones_like(all_distances_pos), np.zeros_like(all_distances_neg)])
    
    threshold = np.mean(all_distances)
    predictions = (all_distances < threshold).astype(int)
    
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    mcc = matthews_corrcoef(all_labels, predictions)
    
    return running_loss / len(dataloader), accuracy, precision, recall, f1, mcc

# Training
print("Starting Training!")
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}")
    train_loss = train_epoch(siamese_net, train_dataloader, triplet_criterion, optimizer, device)
    print(f"Completed training for epoch {epoch+1}")
    
    val_loss, accuracy, precision, recall, f1, mcc = evaluate(siamese_net, val_dataloader, triplet_criterion, device)
    
    print(f'Epoch {epoch+1}:')
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}')

    # Optionally, save the model
    if (epoch + 1) % 10 == 0:
        model_save_path = f"{current_dir}/siamese_model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': siamese_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, model_save_path)

print("Training completed!")
