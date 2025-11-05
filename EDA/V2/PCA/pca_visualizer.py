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
    """Generate heatmap of PCA loadings."""
    n_show = min(n_show, loadings_df.shape[1])

    fig, ax = plt.subplots(figsize=(12, max(8, len(loadings_df) * 0.3)))

    short_names = [name[:30] + '...' if len(name) > 30 else name
                   for name in loadings_df.index]
    loadings_plot = loadings_df.iloc[:, :n_show].copy()
    loadings_plot.index = short_names

    sns.heatmap(loadings_plot, annot=False, cmap="RdBu_r",
                center=0, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f'PCA Loadings Heatmap (Top {n_show} Components)',
                 fontsize=14, pad=15)
    ax.set_xlabel('Principal Components', fontsize=11)
    ax.set_ylabel('Features', fontsize=11)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loadings_heatmap.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("- Loadings heatmap saved")


def save_pca_results(loadings_df, pca_df, save_dir):
    """Save loadings and transformed data to CSV."""
    loadings_df.to_csv(os.path.join(save_dir, "pca_loadings.csv"))
    pca_df.to_csv(os.path.join(save_dir, "pca_transformed_data.csv"))
    print(f"- Results saved to: {save_dir}")
