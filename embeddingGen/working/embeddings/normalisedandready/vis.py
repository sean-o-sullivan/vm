import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from tqdm import tqdm
import umap.umap_ as umap

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
        self.value_columns = list(self.data.columns[2:-1])  
        self.core_info_column = self.data.columns[-1]  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        embedding = row[self.value_columns].values.astype(float)  
        core_info = row[self.core_info_column]
        return torch.tensor(embedding, dtype=torch.float32), core_info

    def preprocess_data(self):
        
        self.data[self.value_columns] = self.data[self.value_columns].apply(pd.to_numeric, errors='coerce')
       
        
        nan_columns = self.data[self.value_columns].columns[self.data[self.value_columns].isna().any()].tolist()
        if nan_columns:
            print(f"Warning: NaN values found in the following columns: {nan_columns}")
            print("These values will be replaced with 0.0")
       
        
        self.data[self.value_columns] = self.data[self.value_columns].fillna(0.0)

    def get_column_info(self):
        return f"Total columns: {len(self.data.columns)}\nEmbedding columns: {len(self.value_columns)}\nExcluded columns: {', '.join(self.data.columns[:2])} and {self.core_info_column}"

def get_core_info_at_depth(core_info_str, depth=1):
    parts = core_info_str.split(' - ')
    if depth > len(parts):
        return parts[-1]
    return parts[depth - 1]

def process_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_core_info = []
   
    with torch.no_grad():
        for embeddings, core_info in tqdm(dataloader, desc="Processing embeddings"):
            embeddings = embeddings.to(device)
            encoded_embeddings = model(embeddings)
            all_embeddings.append(encoded_embeddings.cpu().numpy())
            all_core_info.extend(core_info)
   
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_core_info

def visualize_embeddings_3d_umap(embeddings, core_infos, depth, output_file, opacity=0.8):
    core_infos_at_depth = [get_core_info_at_depth(info, depth) for info in core_infos]
    reducer = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d = reducer.fit_transform(embeddings)
    unique_core_infos = list(set(core_infos_at_depth))
    color_palette = [
        '
        '
        '
        '
    ]
    color_dict = {core_info: color_palette[i % len(color_palette)] for i, core_info in enumerate(unique_core_infos)}
    fig = go.Figure()
   
    for core_info in unique_core_infos:
        core_info_mask = np.array(core_infos_at_depth) == core_info
        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[core_info_mask, 0],
            y=embeddings_3d[core_info_mask, 1],
            z=embeddings_3d[core_info_mask, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=color_dict[core_info],
                opacity=opacity
            ),
            text=[f"Core Info: {core_info}" for _ in range(sum(core_info_mask))],
            hoverinfo='text',
            name=f"Core Info: {core_info}",
            customdata=np.array([[core_info]] * sum(core_info_mask)),
            hoverlabel=dict(namelength=-1)
        ))
   
    fig.update_layout(
        title=f"3D UMAP visualization at Core Info Depth {depth}",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            xaxis=dict(showgrid=False, backgroundcolor='white'),
            yaxis=dict(showgrid=False, backgroundcolor='white'),
            zaxis=dict(showgrid=False, backgroundcolor='white')
        ),
        width=1500,
        height=800,
        hovermode="closest",
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Reset View",
                method="relayout",
                args=[{"scene.camera": dict(eye=dict(x=1.25, y=1.25, z=1.25))}]
            )]
        )]
    )

    fig.update_layout(
        annotations=[dict(
            text="Hover over points to see core info",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False
        )]
    )

    config = {'responsive': True}
    fig.write_html(output_file, include_plotlyjs='cdn', config=config)
    print(f"Interactive 3D UMAP visualization saved to {output_file}")
    fig.show()

if __name__ == "__main__":
    
    hidden_size = 256
    batch_size = 128

    
    current_dir = os.getcwd()
    dataset = EmbeddingDataset(os.path.join(current_dir, "GG_100_updated_core_info_only.csv"))
    dataset.preprocess_data()  
   
    
    print(dataset.get_column_info())
   
    
    input_size = len(dataset.value_columns)
    print(f"Input size: {input_size}")

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    
    model_path = os.path.join(current_dir, "BnG_10_best_transformer_siamese_model.pth")
    checkpoint = torch.load(model_path, map_location=device)

    siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)
    siamese_net.load_state_dict(checkpoint['model_state_dict'])
    siamese_net.eval()

    print("Starting Embedding Processing...")

    
    embeddings, core_infos = process_embeddings(siamese_net, dataloader, device)

    
    visualize_embeddings_3d_umap(embeddings, core_infos, depth=2, output_file="core_info_embeddings_3d_umap.html", opacity=1)

    print(f"\nProcessing completed! UMAP Visualization saved to core_info_embeddings_3d_umap.html")