import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DPI

sns.set_theme(style="whitegrid")


def plot_scree(pca_full, cumulative_variance, n_components, variance_threshold, save_dir):
    """Generate scree plot with individual and cumulative variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    ax1.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_,
            alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Variance Explained Ratio', fontsize=11)
    ax1.set_title('Scree Plot - Individual Variance', fontsize=13, pad=10)
    ax1.grid(alpha=0.3)

    # Cumulative variance
    ax2.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, marker='o', linestyle='-',
             color='darkgreen', linewidth=2, markersize=5)
    ax2.axhline(y=variance_threshold, color='r', linestyle='--',
                label=f'{variance_threshold * 100:.0f}% threshold')
    ax2.axvline(x=n_components, color='orange', linestyle='--',
                label=f'{n_components} components')
    ax2.set_xlabel('Number of Components', fontsize=11)
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=11)
    ax2.set_title('Cumulative Variance Explained', fontsize=13, pad=10)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scree_plot.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("- Scree plot saved")


def plot_loadings_heatmap(loadings_df, save_dir, n_show=10):
    """Plot heatmap of PCA loadings for top contributors."""
    # Select top N features by absolute loading on PC1
    top_features = loadings_df.abs()['PC1'].nlargest(n_show).index
    loadings_subset = loadings_df.loc[top_features]

    # Dynamic figure size based on number of features and components
    n_features = len(top_features)
    n_components = loadings_df.shape[1]
    fig_width = max(12, n_components * 1.5)  # Minimum 12, grows with components
    fig_height = max(8, n_features * 0.6)  # Minimum 8, grows with features

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(loadings_subset, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, cbar_kws={'label': 'Loading'},
                annot_kws={"size": 9}, ax=ax, linewidths=0.5)

    ax.set_title(f'PCA Loadings - Top {n_show} Features', fontsize=14, pad=15)
    ax.set_xlabel('Principal Components', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)

    # Improve label readability
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_loadings_heatmap.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("- PCA loadings heatmap saved")


def save_pca_results(loadings_df, pca_df, save_dir):
    """Save loadings and transformed data to CSV."""
    loadings_df.to_csv(os.path.join(save_dir, "pca_loadings.csv"))
    pca_df.to_csv(os.path.join(save_dir, "pca_transformed_data.csv"))
    print(f"- Results saved to: {save_dir}")
