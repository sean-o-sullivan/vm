import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, matthews_corrcoef
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import optuna
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_heads=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.norm = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(0)  # Add sequence dimension
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Remove sequence dimension
        x = self.output_proj(x)
        x = self.norm(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization

class SiameseTransformerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(SiameseTransformerNetwork, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size, num_layers, num_heads, dropout)

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

def train_epoch(siamese_model, dataloader, criterion, optimizer, device, epoch, trial):
    siamese_model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    for batch_idx, (anchor, positive, negative) in enumerate(pbar):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        
        anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
        
        dist_pos = F.pairwise_distance(anchor_out, positive_out)
        dist_neg = F.pairwise_distance(anchor_out, negative_out)
        
        target = torch.ones(anchor_out.size(0)).to(device)
        
        loss = criterion(dist_neg, dist_pos, target)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
        
        if batch_idx % 10 == 0:
            trial.report(running_loss / (batch_idx + 1), epoch * len(dataloader) + batch_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return running_loss / len(dataloader)

def find_best_threshold(distances, labels):
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def evaluate(siamese_model, dataloader, criterion, device):
    siamese_model.eval()
    all_positive_distances = []
    all_negative_distances = []
    
    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_out, positive_out, negative_out = siamese_model(anchor, positive, negative)
            
            dist_pos = F.pairwise_distance(anchor_out, positive_out)
            dist_neg = F.pairwise_distance(anchor_out, negative_out)
            
            all_positive_distances.extend(dist_pos.cpu().numpy())
            all_negative_distances.extend(dist_neg.cpu().numpy())
            
            all_labels.extend([1] * len(dist_pos) + [0] * len(dist_neg))
    
    all_distances = np.concatenate([all_positive_distances, all_negative_distances])
    all_labels = np.concatenate([np.ones(len(all_positive_distances)), np.zeros(len(all_negative_distances))])
    
    best_threshold = find_best_threshold(all_distances, all_labels)
    predictions = (all_distances < best_threshold).astype(int)
    mcc = matthews_corrcoef(all_labels, predictions)
    
    auc_score = auc(all_labels, -all_distances)
    
    mean_pos_dist = np.mean(all_positive_distances)
    mean_neg_dist = np.mean(all_negative_distances)
    
    return mean_pos_dist, mean_neg_dist, mcc, auc_score, best_threshold

def objective(trial):
    input_size = 112 
    hidden_size = 256 
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    margin = trial.suggest_float('margin', 0.1, 2.0)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    
    batch_size = 128
    num_epochs = 30  # Reduced for faster trials

    siamese_net = SiameseTransformerNetwork(input_size, hidden_size, num_layers, num_heads, dropout).to(device)
    criterion = nn.MarginRankingLoss(margin=margin)
    optimizer = optim.Adam(siamese_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    current_dir = os.getcwd()
    train_dataset = TripletDataset(os.path.join(current_dir, "BnG_70.csv"))
    val_dataset = TripletDataset(os.path.join(current_dir, "BnG_30.csv"))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    best_mcc = float('-inf')

    for epoch in range(num_epochs):
        train_loss = train_epoch(siamese_net, train_dataloader, criterion, optimizer, device, epoch, trial)
        scheduler.step()
        
        mean_pos_dist, mean_neg_dist, mcc, auc_score, best_threshold = evaluate(siamese_net, val_dataloader, criterion, device)
        
        logging.info(f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs}")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  MCC: {mcc:.4f}, AUC: {auc_score:.4f}")
        logging.info(f"  Mean Pos Dist: {mean_pos_dist:.4f}, Mean Neg Dist: {mean_neg_dist:.4f}")
        logging.info(f"  Best Threshold: {best_threshold:.4f}")
        
        if mcc > best_mcc:
            best_mcc = mcc

        trial.report(mcc, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_mcc

if __name__ == "__main__":
    study_name = f"siamese_transformer_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    )
    
    study.optimize(objective, n_trials=50, timeout=3600*6)  # 6 hours timeout

    logging.info("Best trial:")
    trial = study.best_trial

    logging.info(f"  Value: {trial.value}")
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")

    # Train the final model with the best hyperparameters
    input_size = 112
    hidden_size = 256
    lr = trial.params['lr']
    margin = trial.params['margin']
    dropout = trial.params['dropout']
    num_layers = trial.params['num_layers']
    num_heads = trial.params['num_heads']
    batch_size = 128
    num_epochs = 100  # Increased for final training

    best_siamese_net = SiameseTransformerNetwork(input_size, hidden_size, num_layers, num_heads, dropout).to(device)
    best_criterion = nn.MarginRankingLoss(margin=margin)
    best_optimizer = optim.Adam(best_siamese_net.parameters(), lr=lr, weight_decay=1e-4)
    best_scheduler = CosineAnnealingLR(best_optimizer, T_max=num_epochs)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    best_mcc = float('-inf')
    best_model_path = None

    logging.info("Starting Final Training!")
    for epoch in range(num_epochs):
        train_loss = train_epoch(best_siamese_net, train_dataloader, best_criterion, best_optimizer, device, epoch, optuna.trial.FixedTrial(trial.params))
        best_scheduler.step()
        
        mean_pos_dist, mean_neg_dist, mcc, auc_score, best_threshold = evaluate(best_siamese_net, val_dataloader, best_criterion, device)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {train_loss:.4f}')
        logging.info(f'Mean Positive Distance: {mean_pos_dist:.4f}')
        logging.info(f'Mean Negative Distance: {mean_neg_dist:.4f}')
        logging.info(f'MCC: {mcc:.4f}')
        logging.info(f'AUC: {auc_score:.4f}')
        logging.info(f'Best Threshold: {best_threshold:.4f}')
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_model_path = f"{current_dir}/BnG_2_best_transformer_siamese_model_mcc.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_siamese_net.state_dict(),
                'optimizer_state_dict': best_optimizer.state_dict(),
                'scheduler_state_dict': best_scheduler.state_dict(),
                'train_loss': train_loss,
                'mcc': mcc,
                'auc': auc_score,
                'best_threshold': best_threshold
            }, best_model_path)
            logging.info(f"New best model found and saved at epoch {epoch+1} with MCC: {mcc:.4f}")
        
        if (epoch + 1) % 10 == 0:
            model_save_path = f"{current_dir}/transformer_siamese_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_siamese_net.state_dict(),
                'optimizer_state_dict': best_optimizer.state_dict(),
                'scheduler_state_dict': best_scheduler.state_dict(),
                'train_loss': train_loss,
                'mcc': mcc,
                'best_threshold': best_threshold
            }, model_save_path)

    logging.info("Training completed!")
    logging.info(f"Best model is now saved at {best_model_path} with MCC: {best_mcc:.4f}")
