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

# Simplified
class SimpleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Simplified Siamese Network
class SimpleSiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleSiameseNetwork, self).__init__()
        self.encoder = SimpleEncoder(input_size, hidden_size)

    def forward(self, anchor, positive, negative):
        anchor_out = self.encoder(anchor)
        positive_out = self.encoder(positive)
        negative_out = self.encoder(negative)
        return anchor_out, positive_out, negative_out

# Simplified Classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim * 3, 1)

    def forward(self, anchor, positive, negative):
        x = torch.cat((anchor, positive, negative), dim=1)
        return self.fc(x).squeeze(-1)

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
hidden_size = 64
lr = 0.001
batch_size = 64
num_epochs = 50

siamese_net = SimpleSiameseNetwork(input_size, hidden_size).to(device)
classifier_net = SimpleClassifier(hidden_size).to(device)

triplet_criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
bce_criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(list(siamese_net.parameters()) + list(classifier_net.parameters()), lr=lr)

current_dir = os.getcwd()
train_dataset = TripletDataset(os.path.join(current_dir, "Final-Triplets_G_70_|_VTL5_C3.csv"))
val_dataset = TripletDataset(os.path.join(current_dir, "Final-Triplets_G_30_|_VTL5_C3.csv"))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

# Training 
def train_epoch(siamese_model, classifier_model, dataloader, triplet_criterion, bce_criterion, optimizer, device):
    siamese_model.train()
    classifier_model.train()
    running_loss = 0.0
    for anchor, positive, negative in dataloader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        
        anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
        triplet_loss = triplet_criterion(anchor_out, positive_out, negative_out)
        
        positive_classifier_out = classifier_model(anchor_out, positive_out, negative_out)
        negative_classifier_out = classifier_model(anchor_out, negative_out, positive_out)
        
        bce_loss_positive = bce_criterion(positive_classifier_out, torch.ones_like(positive_classifier_out))
        bce_loss_negative = bce_criterion(negative_classifier_out, torch.zeros_like(negative_classifier_out))
        
        loss = triplet_loss + bce_loss_positive + bce_loss_negative
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Evaluation 
def evaluate(siamese_model, classifier_model, dataloader, triplet_criterion, bce_criterion, device):
    siamese_model.eval()
    classifier_model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            triplet_loss = triplet_criterion(anchor_out, positive_out, negative_out)
            
            positive_classifier_out = classifier_model(anchor_out, positive_out, negative_out)
            negative_classifier_out = classifier_model(anchor_out, negative_out, positive_out)
            
            bce_loss_positive = bce_criterion(positive_classifier_out, torch.ones_like(positive_classifier_out))
            bce_loss_negative = bce_criterion(negative_classifier_out, torch.zeros_like(negative_classifier_out))
            
            loss = triplet_loss + bce_loss_positive + bce_loss_negative
            running_loss += loss.item()
            
            predictions_positive = (torch.sigmoid(positive_classifier_out) > 0.5).float().cpu().numpy()
            predictions_negative = (torch.sigmoid(negative_classifier_out) <= 0.5).float().cpu().numpy()
            
            all_predictions.extend(predictions_positive)
            all_labels.extend(np.ones_like(predictions_positive))
            all_predictions.extend(predictions_negative)
            all_labels.extend(np.zeros_like(predictions_negative))
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    
    label_0_accuracy = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0
    label_1_accuracy = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0
    
    return running_loss / len(dataloader), overall_accuracy, label_0_accuracy, label_1_accuracy, precision, recall, f1, mcc

# Training loop
print("Starting Training!")
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}")
    train_loss = train_epoch(siamese_net, classifier_net, train_dataloader, triplet_criterion, bce_criterion, optimizer, device)
    print(f"Completed training for epoch {epoch+1}")
    
    val_loss, overall_accuracy, label_0_accuracy, label_1_accuracy, precision, recall, f1, mcc = evaluate(siamese_net, classifier_net, val_dataloader, triplet_criterion, bce_criterion, device)
    
    print(f'Epoch {epoch+1}:')
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {overall_accuracy:.4f}')
    print(f'Label 0 Accuracy: {label_0_accuracy:.4f}, Label 1 Accuracy: {label_1_accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}')

print("Training is completed!")
