import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

input_size = 112
hidden_size = 256
lr = 0.001
batch_size = 128
num_epochs = 200

siamese_net = EnhancedSiameseNetwork(input_size, hidden_size).to(device)

# Use MarginRankingLoss
margin = 0.3
criterion = nn.MarginRankingLoss(margin=margin)
optimizer = optim.Adam(siamese_net.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

def evaluate(siamese_model, dataloader, criterion, device):
    siamese_model.eval()
    running_loss = 0.0
    all_distances = []
    all_labels = []
    
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
            
            all_distances.extend(dist_pos.cpu().numpy())
            all_distances.extend(dist_neg.cpu().numpy())
            all_labels.extend(np.ones(anchor_out.size(0)))
            all_labels.extend(np.zeros(anchor_out.size(0)))
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)
    
    # distance-based metrics
    mean_pos_dist = np.mean(all_distances[all_labels == 1])
    mean_neg_dist = np.mean(all_distances[all_labels == 0])
    std_pos_dist = np.std(all_distances[all_labels == 1])
    std_neg_dist = np.std(all_distances[all_labels == 0])
    
    # overlap
    pos_distances = all_distances[all_labels == 1]
    neg_distances = all_distances[all_labels == 0]
    overlap_min = max(np.min(pos_distances), np.min(neg_distances))
    overlap_max = min(np.max(pos_distances), np.max(neg_distances))
    overlap_range = max(0, overlap_max - overlap_min)
    total_range = max(np.max(pos_distances), np.max(neg_distances)) - min(np.min(pos_distances), np.min(neg_distances))
    overlap_percentage = (overlap_range / total_range) * 100 if total_range > 0 else 0
    
    # AUC
    auc = roc_auc_score(all_labels, -all_distances)  # Negative because smaller distance = more similar
    
    # accuracy using the mean distance as threshold
    threshold = np.mean(all_distances)
    predictions = (all_distances < threshold).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    
    return (running_loss / len(dataloader), mean_pos_dist, mean_neg_dist, std_pos_dist, std_neg_dist, 
            overlap_percentage, auc, accuracy, threshold)

# Best model tracking
best_accuracy = float('-inf')
best_model_path = None

print("Starting Training!")
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}")
    
    train_loss = train_epoch(siamese_net, train_dataloader, criterion, optimizer, device)
    scheduler.step()
    
    val_loss, mean_pos_dist, mean_neg_dist, std_pos_dist, std_neg_dist, overlap_percentage, auc, accuracy, threshold = evaluate(siamese_net, val_dataloader, criterion, device)
    
    print(f'Epoch {epoch+1}:')
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Mean Positive Distance: {mean_pos_dist:.4f} ± {std_pos_dist:.4f}')
    print(f'Mean Negative Distance: {mean_neg_dist:.4f} ± {std_neg_dist:.4f}')
    print(f'Distance Difference: {mean_neg_dist - mean_pos_dist:.4f}')
    print(f'Overlap Percentage: {overlap_percentage:.2f}%')
    print(f'AUC: {auc:.4f}')
    print(f'Accuracy: {accuracy:.4f} (Threshold: {threshold:.4f})')
    
    # Check if this is the best model so far based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = f"{current_dir}/best_distance_siamese_model.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': siamese_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'auc': auc,
            'overlap_percentage': overlap_percentage,
            'threshold': threshold
        }, best_model_path)
        print(f"New best model found and saved at epoch {epoch+1} with Accuracy: {accuracy:.4f}")
    
    # Regular saving every 10 epochs
    if (epoch + 1) % 10 == 0:
        model_save_path = f"{current_dir}/distance_siamese_model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': siamese_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'threshold': threshold
        }, model_save_path)

print("Training completed!")
print(f"Best model saved at {best_model_path} with Accuracy: {best_accuracy:.4f}")
