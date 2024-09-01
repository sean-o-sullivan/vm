import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
import os
import random
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureAwareTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(FeatureAwareTransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class EnhancedEmbeddingNet(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1):
        super(EnhancedEmbeddingNet, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.transformer_layers = nn.ModuleList([
            FeatureAwareTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # sequence dimension
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.squeeze(1) 
        x = self.output_proj(x)
        x = self.layer_norm(x)
        return x

class EnhancedSiameseNetwork(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(EnhancedSiameseNetwork, self).__init__()
        self.embedding_net = EnhancedEmbeddingNet(input_size, d_model, nhead, num_layers, dim_feedforward, dropout)

    def forward(self, anchor, positive, negative):
        output_anchor = self.embedding_net(anchor)
        output_positive = self.embedding_net(positive)
        output_negative = self.embedding_net(negative)
        return output_anchor, output_positive, output_negative

class EnhancedClassifierNet(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedClassifierNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim*3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
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
d_model = 128
nhead = 8
num_layers = 6
dim_feedforward = 512
dropout = 0.1
lr = 0.0001
batch_size = 64
num_epochs = 100

siamese_net = EnhancedSiameseNetwork(input_size, d_model, nhead, num_layers, dim_feedforward, dropout).to(device)
classifier_net = EnhancedClassifierNet(d_model).to(device)



# Optimizer and loss
optimizer = optim.AdamW(list(siamese_net.parameters()) + list(classifier_net.parameters()), lr=lr, weight_decay=1e-5)
triplet_criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
bce_criterion = nn.BCEWithLogitsLoss()

current_dir = os.getcwd()
train_dataset = TripletDataset(os.path.join(current_dir, "Final-Triplets_G_70_|_VTL5_C3.csv"))
val_dataset = TripletDataset(os.path.join(current_dir, "Final-Triplets_G_30_|_VTL5_C3.csv"))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

def train_epoch(siamese_model, classifier_model, dataloader, triplet_criterion, bce_criterion, optimizer, device):
    siamese_model.train()
    classifier_model.train()
    running_loss = 0.0
    total_batches = len(dataloader)
    for i, (anchor, positive, negative) in enumerate(dataloader, start=1):
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
        print(f'Training {i}/{total_batches}', end='\r')
    return running_loss / total_batches
    



def evaluate(siamese_model, classifier_model, dataloader, triplet_criterion, bce_criterion, device):
    siamese_model.eval()
    classifier_model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for i, (anchor, positive, negative) in enumerate(dataloader, start=1):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            # Forward pass 
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            triplet_loss = triplet_criterion(anchor_out, positive_out, negative_out)
            
            # Forward pass through the classifier 
            positive_classifier_out = classifier_model(anchor_out, positive_out, negative_out)
            negative_classifier_out = classifier_model(anchor_out, negative_out, positive_out)
            
            # Calculate BCE losses
            bce_loss_positive = bce_criterion(positive_classifier_out, torch.ones_like(positive_classifier_out))
            bce_loss_negative = bce_criterion(negative_classifier_out, torch.zeros_like(negative_classifier_out))
            
            # Total loss
            loss = triplet_loss + bce_loss_positive + bce_loss_negative
            running_loss += loss.item()
            
            # Sigmoid outputs for predictions
            predictions_positive = (torch.sigmoid(positive_classifier_out) > 0.5).float().cpu().numpy()
            predictions_negative = (torch.sigmoid(negative_classifier_out) > 0.5).float().cpu().numpy()
            
            all_predictions.extend(predictions_positive)
            all_labels.extend(np.ones_like(predictions_positive))

            all_predictions.extend(predictions_negative)
            all_labels.extend(np.zeros_like(predictions_negative))  

            print(f'Validation {i}/{total_batches}', end='\r')
    
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate evaluation metrics
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    
    if cm.shape[0] > 1:
        label_0_accuracy = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0
        label_1_accuracy = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0
    else:
        label_0_accuracy = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0
        label_1_accuracy = 0
    
    return running_loss / total_batches, overall_accuracy, label_0_accuracy, label_1_accuracy, precision, recall, f1, mcc





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

    # Debug prints
    with torch.no_grad():
        for i, (anchor, positive, negative) in enumerate(val_dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_out, positive_out, negative_out = siamese_net(anchor, positive, negative)
            positive_classifier_out = classifier_net(anchor_out, positive_out, negative_out)
            negative_classifier_out = classifier_net(anchor_out, negative_out, positive_out)
            print(f"Positive outputs: {torch.sigmoid(positive_classifier_out[:5])}")
            print(f"Negative outputs: {torch.sigmoid(negative_classifier_out[:5])}")
            break

    # Optionally, save
    if (epoch + 1) % 10 == 0:
        model_save_path = f"{current_dir}/siamese_model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'siamese_model_state_dict': siamese_net.state_dict(),
            'classifier_model_state_dict': classifier_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, model_save_path)

print("Training completed!")
