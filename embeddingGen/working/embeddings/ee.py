import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import seaborn as sns


def load_and_combine_data(file_paths):
    dataframes = [pd.read_csv(file_path) for file_path in file_paths]
    return pd.concat(dataframes, ignore_index=True)

def calculate_discriminative_score(cv, kurtosis, skewness):
    return (np.abs(cv) + np.abs(kurtosis) + np.abs(skewness)) / 3

def visualize_features(combined_df, stats_data, output_file, dpi=300, discriminative_threshold=0.5):
    numeric_columns = [col for col in combined_df.columns if col != 'embedding_id' and pd.api.types.is_numeric_dtype(combined_df[col])]
    n_features = len(numeric_columns)
    n_cols = 5
    n_rows = (n_features - 1) // n_cols + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    fig.suptitle("Feature Distributions and Z-Scores", fontsize=16, y=1.02)

    for idx, col in enumerate(numeric_columns):
        ax = axes[idx // n_cols, idx % n_cols]
        
        mean = stats_data.at[col, 'mean']
        std = stats_data.at[col, 'std']
        all_zeros = stats_data.at[col, 'all_zeros']
        cv = stats_data.at[col, 'cv']
        kurtosis = stats_data.at[col, 'kurtosis']
        skewness = stats_data.at[col, 'skewness']
        
        discriminative_score = calculate_discriminative_score(cv, kurtosis, skewness)
        is_discriminative = discriminative_score > discriminative_threshold

        if all_zeros:
            ax.text(0.5, 0.5, "ALL ZEROS", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
        else:
            z_scores = (combined_df[col] - mean) / std if std != 0 else np.zeros_like(combined_df[col])
            
            n, bins, _ = ax.hist(z_scores, bins=50, density=True, alpha=0.7)
            
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, 0, 1)
            ax2 = ax.twinx()
            ax2.plot(x, p, 'r-', linewidth=2)
            
            ax2.set_ylabel("PDF")

        ax.set_title(f"{col}")
        ax.set_xlabel("Z-Score")
        ax.set_ylabel("Frequency")
        
        discriminative_text = f"Discriminative" if is_discriminative else "Not Discriminative"
        discriminative_color = "green" if is_discriminative else "red"
        ax.text(0.5, -0.15, f"{discriminative_text}\nScore: {discriminative_score:.2f}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color=discriminative_color)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    return stats_data



def calculate_statistics(combined_df):
    numeric_columns = combined_df.select_dtypes(include='number').columns
    numeric_columns = [col for col in numeric_columns if col != 'embedding_id']
    stats_dict = {}

    for col in numeric_columns:
        data = combined_df[col]
        mean = data.mean()
        std = data.std()
        cv = std / mean if mean != 0 else 0
        stats_dict[col] = {
            'mean': mean,
            'std': std,
            'percentile_99.5': np.percentile(data, 99.5),
            'percentile_0.5': np.percentile(data, 0.5),
            'all_zeros': np.all(data == 0),
            'cv': cv,
            'kurtosis': stats.kurtosis(data),
            'skewness': stats.skew(data)
        }

    return pd.DataFrame(stats_dict).T

def normalize_and_filter_embeddings(csv_of_embeddings, stats_data, features_to_omit, output_dir):
    raw_embedding = pd.read_csv(csv_of_embeddings)
    
    normalized_embedding = pd.DataFrame()
    if 'embedding_id' in raw_embedding.columns:
        normalized_embedding['embedding_id'] = raw_embedding['embedding_id']
    if 'author_id' in raw_embedding.columns:
        normalized_embedding['author_id'] = raw_embedding['author_id']
    
    for col in raw_embedding.columns:
        if col not in ['embedding_id', 'author_id'] and col in stats_data.index and col not in features_to_omit:
            mean = stats_data.at[col, 'mean']
            std = stats_data.at[col, 'std']
            percentile_99_5 = stats_data.at[col, 'percentile_99.5']
            percentile_0_5 = stats_data.at[col, 'percentile_0.5']
            
            if std != 0:
                z_scores = (raw_embedding[col] - mean) / std
                
                # Calculate z-scores for 0.5th and 99.5th percentiles
                z_score_0_5 = (percentile_0_5 - mean) / std
                z_score_99_5 = (percentile_99_5 - mean) / std
                
                # Linear mapping of z-scores to [0, 1] range
                normalized_value = (z_scores - z_score_0_5) / (z_score_99_5 - z_score_0_5)
                
                # Clip values to ensure they are between 0 and 1
                normalized_embedding[col] = np.clip(normalized_value, 0, 1)
            else:
                # If std is 0, set all values to 0.5
                normalized_embedding[col] = 0.5
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = os.path.basename(csv_of_embeddings)
    output_file_path = os.path.join(output_dir, f"normalized_{file_name}")
    
    normalized_embedding.to_csv(output_file_path, index=False)
    
    print(f"Normalized embedding saved to: {output_file_path}")




def normalize_future_csv(csv_path, stats_data_path, features_to_omit, output_dir):
    """
    Normalize a future CSV file using pre-calculated statistics.
    
    Parameters:
    - csv_path: Path to the CSV file to be normalized
    - stats_data_path: Path to the pre-calculated statistics CSV file
    - features_to_omit: List of features to exclude from normalization
    - output_dir: Directory to save the normalized CSV
    """
    stats_data = pd.read_csv(stats_data_path, index_col=0)
    normalize_and_filter_embeddings(csv_path, stats_data, features_to_omit, output_dir)


def visualize_normalized_data(normalized_file_paths, output_file, dpi=300):
    normalized_dfs = [pd.read_csv(file_path) for file_path in normalized_file_paths]
    combined_df = pd.concat(normalized_dfs, ignore_index=True)
    feature_columns = [col for col in combined_df.columns if col not in ['embedding_id', 'author_id']]
    n_features = len(feature_columns)
    n_cols = 5
    n_rows = (n_features - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    fig.suptitle("Distribution of Normalized Features", fontsize=16, y=1.02)
    for idx, feature in enumerate(feature_columns):
        ax = axes[idx // n_cols, idx % n_cols]
        sns.violinplot(data=combined_df[feature], ax=ax)
        
        ax.set_title(feature)
        ax.set_ylim(-0.1, 1.1)  
        ax.set_ylabel("Normalized Value")
        ax.set_xticks([])

    for idx in range(n_features, n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols, idx % n_cols])

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

def main():
    file_paths = [
        'ABB_30_embeddings.csv',
        'ABB_70_embeddings.csv',
        'AGG_30_embeddings.csv',
        'AGG_70_embeddings.csv'
    ]

    combined_df = load_and_combine_data(file_paths)
    stats_data = calculate_statistics(combined_df)
    stats_data.to_csv('embedding_stats.csv', index=True)
    stats_data = visualize_features(combined_df, stats_data, 'feature_distributions.png', dpi=300, discriminative_threshold=0.5)
    output_dir = 'normalisedandready'
    features_to_omit = [
        "ratio_of_sentence_initial_conjunctions",
        "normalized_assonance"
    ]
    
    for csv_file in file_paths:
        normalize_and_filter_embeddings(csv_file, stats_data, features_to_omit, output_dir)
    
    normalized_file_paths = [os.path.join(output_dir, f"normalized_{os.path.basename(file)}") for file in file_paths]
    visualize_normalized_data(normalized_file_paths, 'normalized_feature_distributions.png', dpi=300)

if __name__ == "__main__":
    main()
