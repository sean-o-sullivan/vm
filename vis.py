import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_heads=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
                                                   dim_feedforward=hidden_size*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.norm = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        x = self.output_proj(x)
        x = self.norm(x)
        return x

class SiameseTransformerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseTransformerNetwork, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size)

    def forward(self, x):
        return self.encoder(x)

class EmbeddingDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.value_columns = [col for col in self.data.columns if col not in ['embedding_id', 'author']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        embedding = row[self.value_columns].values.tolist()
        author = row['author']
        return torch.tensor(embedding, dtype=torch.float32), author

def process_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_authors = []
    with torch.no_grad():
        for embeddings, authors in dataloader:
            embeddings = embeddings.to(device)
            encoded_embeddings = model(embeddings)
            all_embeddings.append(encoded_embeddings.cpu().numpy())
            all_authors.extend(authors)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_authors

def visualize_embeddings_3d(embeddings, authors):
    pass

input_size = 112
hidden_size = 256
batch_size = 128

current_dir = os.getcwd()
model_path = os.path.join(current_dir, "model_checkpoint.pth")
siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)

dataset = EmbeddingDataset(os.path.join(current_dir, "embedding_data.csv"))
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

print("Processing embeddings...")
embeddings, authors = process_embeddings(siamese_net, dataloader, device)

print("Embedding processing complete. Visualization not yet implemented.")
