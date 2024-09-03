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
        self.pos_encoder = nn.Embedding(1, hidden_size)  # Simple positional encoding
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(0)  # sequence dimension
        pos = self.pos_encoder(torch.zeros(1, device=x.device, dtype=torch.long))
        x = x + pos
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Remove sequence dimension
        x = self.output_proj(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization

class TransformerSiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TransformerSiameseNetwork, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size)

    def forward(self, anchor, comparison):
        anchor_out = self.encoder(anchor)
        comparison_out = self.encoder(comparison)
        return anchor_out, comparison_out

class AuthorshipDataset(Dataset):
    def __init__(self, csv_file, column):
        self.data = pd.read_csv(csv_file)
        self.column = column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        anchor_embedding = ast.literal_eval(row['anchor_embedding'])
        comparison_embedding = ast.literal_eval(row[self.column])
        label = 1 if row['same_author'] else 0
        
        return (torch.tensor(anchor_embedding, dtype=torch.float32),
                torch.tensor(comparison_embedding, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for anchor, comparison, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            anchor, comparison, labels = anchor.to(device), comparison.to(device), labels.to(device)
            
            optimizer.zero_grad()
            anchor_out, comparison_out = model(anchor, comparison)
            loss = criterion(anchor_out, comparison_out, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, comparison, labels in val_loader:
                anchor, comparison, labels = anchor.to(device), comparison.to(device), labels.to(device)
                anchor_out, comparison_out = model(anchor, comparison)
                loss = criterion(anchor_out, comparison_out, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_transformer_siamese_model.pth')

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_distances = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for anchor, comparison, labels in tqdm(dataloader, desc="Evaluating"):
            anchor, comparison = anchor.to(device), comparison.to(device)
            
            anchor_out, comparison_out = model(anchor, comparison)
            
            dist = F.pairwise_distance(anchor_out, comparison_out)
            
            all_distances.extend(dist.cpu().numpy())
            all_predictions.extend((dist < threshold).cpu().numpy().astype(int))
            all_labels.extend(labels.cpu().numpy())
    
    return all_distances, all_predictions, all_labels

input_size = 112
hidden_size = 256
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# model
siamese_net = TransformerSiameseNetwork(input_size, hidden_size).to(device)
criterion = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(siamese_net.parameters(), lr=learning_rate)

# data
train_dataset = AuthorshipDataset('BnG_2_70.csv', 'comparison_embedding_column')
val_dataset = AuthorshipDataset('BnG_2_30.csv', 'comparison_embedding_column')
test_dataset = AuthorshipDataset('BnG_2_30.csv', 'comparison_embedding_column')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Train 
train_model(siamese_net, train_loader, val_loader, criterion, optimizer, num_epochs, device)

# Load the best model for evaluation
checkpoint = torch.load('best_transformer_siamese_model.pth')
siamese_net.load_state_dict(checkpoint['model_state_dict'])

# Evaluate the model
distances, predictions, labels = evaluate_model(siamese_net, test_loader, device)

# Calculate metrics
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save detailed results
results_file = f"transformer_authorship_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(results_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Distance', 'Prediction', 'True Label'])
    for dist, pred, label in zip(distances, predictions, labels):
        csvwriter.writerow([dist, pred, label])

print(f"Detailed results saved to {results_file}")