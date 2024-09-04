import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_heads=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.norm = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        x = self.output_proj(x)
        x = self.norm(x)
        return F.normalize(x, p=2, dim=1)

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
        for embeddings, authors in tqdm(dataloader, desc="Processing embeddings"):
            embeddings = embeddings.to(device)
            encoded_embeddings = model(embeddings)
            all_embeddings.append(encoded_embeddings.cpu().numpy())
            all_authors.extend(authors)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_authors

def visualize_embeddings_3d(embeddings, authors, output_file):
    tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
    embeddings_3d = tsne.fit_transform(embeddings)

    # Create a DataFrame
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'author': authors
    })

    # Get unique authors and assign colors
    unique_authors = df['author'].unique()
    colorscale = px.colors.qualitative.Plotly[:len(unique_authors)]
    color_map = dict(zip(unique_authors, colorscale))

    # Create the 3D scatter plot with Plotly
    fig = go.Figure()

    for author in unique_authors:
        author_data = df[df['author'] == author]
        fig.add_trace(go.Scatter3d(
            x=author_data['x'],
            y=author_data['y'],
            z=author_data['z'],
            mode='markers',
            marker=dict(
                size=3,
                color=color_map[author],
                opacity=0.8
            ),
            name=author,
            hoverinfo='name',
            showlegend=True
        ))

    # Update the layout
    fig.update_layout(
        title="3D t-SNE visualization of author embeddings",
        scene=dict(
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            zaxis_title="t-SNE 3",
            aspectmode='cube'
        ),
        width=1000,
        height=1000,
        hovermode="closest"
    )

    # Add reset view button
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Reset View",
                        method="relayout",
                        args=[{"scene.camera": dict(eye=dict(x=1.25, y=1.25, z=1.25))}]
                    )
                ]
            )
        ]
    )

    # Add custom JavaScript for hover effects
    hover_js = """
    var plotly_div = document.getElementById('plotly');
    var original_opacities = {};
    var highlighted_author = null;

    plotly_div.on('plotly_hover', function(data) {
        var curve = data.points[0].curveNumber;
        var author = data.points[0].data.name;

        if (author !== highlighted_author) {
            highlighted_author = author;

            var update = {
                'marker.opacity': plotly_div.data.map(function(trace, i) {
                    if (typeof original_opacities[i] === 'undefined') {
                        original_opacities[i] = trace.marker.opacity;
                    }
                    return trace.name === author ? original_opacities[i] : 0.1;
                })
            };

            Plotly.restyle('plotly', update);
        }
    });

    plotly_div.on('plotly_unhover', function(data) {
        highlighted_author = null;
        var update = {
            'marker.opacity': Object.values(original_opacities)
        };
        Plotly.restyle('plotly', update);
    });
    """
    config = {'responsive': True}
    fig.write_html(output_file, include_plotlyjs='cdn', config=config, post_script=hover_js, full_html=False, div_id='plotly')
    print(f"Interactive 3D visualization saved to {output_file}")
    fig.show()

input_size = 112
hidden_size = 256
batch_size = 128

current_dir = os.getcwd()
model_path = os.path.join(current_dir, "BnG_9_best_transformer_siamese_model.pth")
checkpoint = torch.load(model_path, map_location=device)
siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)
siamese_net.load_state_dict(checkpoint['model_state_dict'])
siamese_net.eval()

dataset = EmbeddingDataset(os.path.join(current_dir, "GG_100.csv"))
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

print("Starting Embedding Processing...")
embeddings, authors = process_embeddings(siamese_net, dataloader, device)

visualize_embeddings_3d(embeddings, authors, "author_embeddings_3d_visualization.html")

