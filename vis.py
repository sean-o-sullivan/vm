import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_heads=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
                                                   dim_feedforward=hidden_size * 4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.norm = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        x = self.output_proj(x)
        x = self.norm(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)

class SiameseTransformerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseTransformerNetwork, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size)

    def forward(self, x):
        return self.encoder(x)

class EmbeddingDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        required_columns = {'embedding_id', 'author'}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        self.value_columns = [col for col in self.data.columns if col not in required_columns]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        embedding = row[self.value_columns].values.astype(np.float32)
        author = row['author']
        return torch.tensor(embedding, dtype=torch.float32), author

def process_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_authors = []
    with torch.no_grad():
        for embeddings, authors in tqdm(dataloader, desc="Processing embeddings"):
            embeddings = embeddings.to(device)
            encoded_embeddings = model(embeddings)
            all_embeddings.append(encoded_embeddings.cpu().numpy())
            all_authors.extend(authors)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_authors

def visualize_embeddings_3d(embeddings, authors, output_file):
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_3d = tsne.fit_transform(embeddings)

    unique_authors = list(set(authors))
    color_dict = {author: f"rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.7)"
                  for author in unique_authors}

    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=[color_dict[author] for author in authors],
            opacity=0.8
        ),
        text=[f"Author: {author}" for author in authors]
    )])

    fig.update_layout(
        title="3D t-SNE visualization of embeddings",
        scene=dict(
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            zaxis_title="t-SNE 3",
        ),
        width=1000,
        height=800,
    )

    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")
    fig.show()

input_size = 112
hidden_size = 256
batch_size = 64

current_dir = os.getcwd()
model_path = os.path.join(current_dir, "model_checkpoint.pth")
siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)
checkpoint = torch.load(model_path, map_location=device)
siamese_net.load_state_dict(checkpoint['model_state_dict'])
siamese_net.eval()

dataset = EmbeddingDataset(os.path.join(current_dir, "embedding_data.csv"))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

print("Processing embeddings...")
embeddings, authors = process_embeddings(siamese_net, dataloader, device)
visualize_embeddings_3d(embeddings, authors, "author_embeddings_3d_visualization.html")

print("Processing completed.")
