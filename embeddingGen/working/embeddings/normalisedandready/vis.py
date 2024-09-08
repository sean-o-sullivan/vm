import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
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

def visualize_embeddings_3d_umap(embeddings, core_infos, authors, depths, output_file, opacity=0.8):
    reducer = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d = reducer.fit_transform(embeddings)
   
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]

    fig = go.Figure()

    unique_authors = list(set(authors))
    author_color_dict = {author: color_palette[i % len(color_palette)]
                         for i, author in enumerate(unique_authors)}
   
    

    author_legend_names = {}
    author_name_counts = {}
    for author in authors:
        author_name_counts[author] = author_name_counts.get(author, 0) + 1
    for author in unique_authors:
        count = author_name_counts[author]
        author_legend_names[author] = author  

    core_info_color_dicts = {}
    for depth in depths:
        core_infos_at_depth = [get_core_info_at_depth(info, depth) for info in core_infos]
        unique_core_infos = list(set(core_infos_at_depth))
        core_info_color_dicts[depth] = {core_info: color_palette[i % len(color_palette)]
                                      for i, core_info in enumerate(unique_core_infos)}

    data_trace_indices = {}
    trace_index = 0
    
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=[author_color_dict.get(a, 'lightgrey') for a in authors],
            opacity=opacity
        ),
        text=[f"Author: {a}<br>Core Info: {ci}" for a, ci in zip(authors, core_infos)],
        hoverinfo='text',
        name="Authors",
        customdata=np.array(list(zip(authors, core_infos))),
        hoverlabel=dict(namelength=-1),
        visible=False
    ))
    data_trace_indices['Author'] = trace_index
    trace_index += 1

    

    for depth in depths:
        core_infos_at_depth = [get_core_info_at_depth(info, depth) for info in core_infos]
        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=[core_info_color_dicts[depth].get(ci, 'lightgrey') for ci in core_infos_at_depth],
                opacity=opacity
            ),
            text=[f"Core Info: {ci}<br>Author: {a}" for ci, a in zip(core_infos_at_depth, authors)],
            hoverinfo='text',
            name=f"Depth {depth}",
            customdata=np.array(list(zip(core_infos_at_depth, authors))),
            hoverlabel=dict(namelength=-1),
            visible=False
        ))
        data_trace_indices[f'Depth {depth}'] = trace_index
        trace_index += 1

    

    legend_trace_indices = {}
    author_legend_indices = []
    for author in unique_authors:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=6, color=author_color_dict[author]),
            name=author_legend_names[author],  

            showlegend=True,
            visible=False  

        ))
        author_legend_indices.append(trace_index)
        trace_index += 1
    legend_trace_indices['Author'] = author_legend_indices

    

    for depth in depths:
        core_infos_at_depth = [get_core_info_at_depth(info, depth) for info in core_infos]
        unique_core_infos = sorted(set(core_infos_at_depth))
        core_info_legend_indices = []
        for core_info in unique_core_infos:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=6, color=core_info_color_dicts[depth][core_info]),
                name=core_info,
                showlegend=True,
                visible=False  

            ))
            core_info_legend_indices.append(trace_index)
            trace_index +=1
        legend_trace_indices[f'Depth {depth}'] = core_info_legend_indices

    

    total_traces = len(fig.data)
    buttons = []

    

    visible_author = [False] * total_traces
    visible_author[data_trace_indices['Author']] = True
    for idx in legend_trace_indices['Author']:
        visible_author[idx] = True
    buttons.append(dict(
        label='Author',
        method='update',
        args=[{'visible': visible_author},
              {'title': '3D UMAP visualization - By Author'}]
    ))

    

    for depth in depths:
        visible = [False] * total_traces
        visible[data_trace_indices[f'Depth {depth}']] = True
        for idx in legend_trace_indices[f'Depth {depth}']:
            visible[idx] = True
        buttons.append(dict(
            label=f'Depth {depth}',
            method='update',
            args=[{'visible': visible},
                  {'title': f'3D UMAP visualization - Depth {depth}'}]
        ))

    

    for idx, vis in enumerate(visible_author):
        fig.data[idx].visible = vis

    

    fig.update_layout(title='3D UMAP visualization - By Author')

    

    updatemenus = [dict(
        active=0,  

        buttons=buttons,
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.05,
        xanchor="left",
        y=1.15,
        yanchor="top",
        bgcolor='rgba(255, 255, 255, 0.7)',
        bordercolor='rgba(0, 0, 0, 0.5)',
        borderwidth=1,
        font=dict(size=12)
    )]

    

    updatemenus.append(dict(
        type="buttons",
        showactive=False,
        buttons=[dict(
            label="Reset View",
            method="relayout",
            args=[{"scene.camera": dict(eye=dict(x=1.25, y=1.25, z=1.25))}]
        )],
        pad={"r": 10, "t": 10},
        x=0.13,
        xanchor="left",
        y=1.15,
        yanchor="top",
        bgcolor='rgba(255, 255, 255, 0.7)',
        bordercolor='rgba(0, 0, 0, 0.5)',
        borderwidth=1,
        font=dict(size=12)
    ))

    

    fig.update_layout(
        updatemenus=updatemenus,
        title=dict(
            text="3D UMAP visualization - By Author",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            xaxis=dict(showgrid=False, backgroundcolor='white'),
            yaxis=dict(showgrid=False, backgroundcolor='white'),
            zaxis=dict(showgrid=False, backgroundcolor='white'),
            domain=dict(x=[0.2, 0.88], y=[0.5, 1])  
        ),
        width=1600,  
        height=1200,  
        hovermode="closest",
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=100, l=0, r=300, b=0),  
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,  
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.1)',
            borderwidth=1,
            font=dict(size=10),
            itemsizing='constant',  
            itemclick=False,  
            valign='top',
            itemdoubleclick=False
        )
    )


    fig.update_layout(
        annotations=[dict(
            text="Hover over points to see details",
            xref="paper", yref="paper",
            x=0.5, y=1.02,
            showarrow=False,
            font=dict(size=14)
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

    dataset.data['author'] = dataset.data['author'].fillna('Unknown')
    authors = dataset.data['author'].tolist()

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    model_path = os.path.join(current_dir, "BnG_10_best_transformer_siamese_model.pth")
    checkpoint = torch.load(model_path, map_location=device)

    siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)
    siamese_net.load_state_dict(checkpoint['model_state_dict'])
    siamese_net.eval()

    print("Starting Embedding Processing...")
    embeddings, core_infos = process_embeddings(siamese_net, dataloader, device)

    visualize_embeddings_3d_umap(
        embeddings,
        core_infos,
        authors,
        depths=[1, 2, 3],
        output_file="corse_info_embeddings_3d_umap_interactive.html",
        opacity=1
    )

    print(f"\nProcessing completed! Interactive UMAP Visualization saved to core_info_embeddings_3d_umap_Ainteractive.html")
