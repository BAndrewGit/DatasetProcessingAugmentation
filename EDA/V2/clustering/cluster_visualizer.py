import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import DPI

sns.set_theme(style="whitegrid")


def plot_silhouette_scores(kmeans_results, save_dir):
    """Plot silhouette scores for different K values."""
    k_values = sorted(kmeans_results.keys())
    silhouette_scores = [kmeans_results[k]['silhouette'] for k in k_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, silhouette_scores, marker='o', linewidth=2,
            markersize=8, color='steelblue')
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score vs Number of Clusters', fontsize=14, pad=15)
    ax.grid(alpha=0.3)

    # Highlight optimal K
    optimal_k = max(k_values, key=lambda k: kmeans_results[k]['silhouette'])
    optimal_score = kmeans_results[optimal_k]['silhouette']
    ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7,
               label=f'Optimal K={optimal_k} (score={optimal_score:.3f})')
    ax.scatter([optimal_k], [optimal_score], color='red', s=150, zorder=5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "silhouette_scores.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("- Silhouette scores plot saved")


def plot_pca_clusters(X_pca, kmeans_labels, gmm_labels, save_dir):
    """Plot clusters in 2D PCA space for both methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # K-Means
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1],
                           c=kmeans_labels, cmap='tab10',
                           alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    ax1.set_xlabel('PC1', fontsize=11)
    ax1.set_ylabel('PC2', fontsize=11)
    ax1.set_title('K-Means Clustering (PCA Space)', fontsize=13, pad=10)
    plt.colorbar(scatter1, ax=ax1, label='Cluster')

    # GMM
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                           c=gmm_labels, cmap='tab10',
                           alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    ax2.set_xlabel('PC1', fontsize=11)
    ax2.set_ylabel('PC2', fontsize=11)
    ax2.set_title('GMM Clustering (PCA Space)', fontsize=13, pad=10)
    plt.colorbar(scatter2, ax=ax2, label='Cluster')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_clusters_comparison.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("- PCA clusters comparison plot saved")


def plot_risk_score_by_cluster(df_kmeans, df_gmm, save_dir, target='Risk_Score'):
    """Plot Risk_Score distribution by cluster for both methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # K-Means - Box plot
    df_kmeans_sorted = df_kmeans.sort_values('cluster_kmeans')
    df_kmeans_sorted.boxplot(column=target, by='cluster_kmeans', ax=axes[0, 0])
    axes[0, 0].set_title('K-Means: Risk Score Distribution by Cluster', fontsize=12)
    axes[0, 0].set_xlabel('Cluster', fontsize=11)
    axes[0, 0].set_ylabel('Risk Score', fontsize=11)
    plt.sca(axes[0, 0])
    plt.xticks(rotation=0)

    # K-Means - Bar plot with means
    kmeans_stats = df_kmeans.groupby('cluster_kmeans')[target].agg(['mean', 'std', 'count'])
    axes[0, 1].bar(kmeans_stats.index, kmeans_stats['mean'],
                   yerr=kmeans_stats['std'], alpha=0.7, color='steelblue',
                   capsize=5)
    axes[0, 1].set_title('K-Means: Mean Risk Score by Cluster', fontsize=12)
    axes[0, 1].set_xlabel('Cluster', fontsize=11)
    axes[0, 1].set_ylabel('Mean Risk Score ± Std', fontsize=11)
    axes[0, 1].grid(alpha=0.3, axis='y')

    # Add count labels on bars
    for idx, (cluster, row) in enumerate(kmeans_stats.iterrows()):
        axes[0, 1].text(idx, row['mean'] + row['std'] + 0.5,
                        f"n={int(row['count'])}", ha='center', fontsize=9)

    # GMM - Box plot
    df_gmm_sorted = df_gmm.sort_values('cluster_gmm')
    df_gmm_sorted.boxplot(column=target, by='cluster_gmm', ax=axes[1, 0])
    axes[1, 0].set_title('GMM: Risk Score Distribution by Cluster', fontsize=12)
    axes[1, 0].set_xlabel('Cluster', fontsize=11)
    axes[1, 0].set_ylabel('Risk Score', fontsize=11)
    plt.sca(axes[1, 0])
    plt.xticks(rotation=0)

    # GMM - Bar plot with means
    gmm_stats = df_gmm.groupby('cluster_gmm')[target].agg(['mean', 'std', 'count'])
    axes[1, 1].bar(gmm_stats.index, gmm_stats['mean'],
                   yerr=gmm_stats['std'], alpha=0.7, color='salmon',
                   capsize=5)
    axes[1, 1].set_title('GMM: Mean Risk Score by Cluster', fontsize=12)
    axes[1, 1].set_xlabel('Cluster', fontsize=11)
    axes[1, 1].set_ylabel('Mean Risk Score ± Std', fontsize=11)
    axes[1, 1].grid(alpha=0.3, axis='y')

    # Add count labels on bars
    for idx, (cluster, row) in enumerate(gmm_stats.iterrows()):
        axes[1, 1].text(idx, row['mean'] + row['std'] + 0.5,
                        f"n={int(row['count'])}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "risk_score_by_cluster.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("- Risk score analysis plot saved")


def plot_cluster_comparison_summary(comparison, kmeans_silhouette, gmm_silhouette, save_dir):
    """Plot comprehensive comparison summary."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Silhouette comparison
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ['K-Means', 'GMM']
    scores = [kmeans_silhouette, gmm_silhouette]
    colors = ['steelblue', 'salmon']
    bars = ax1.bar(methods, scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Silhouette Score', fontsize=11)
    ax1.set_title('Clustering Quality Comparison', fontsize=12, pad=10)
    ax1.set_ylim([0, max(scores) * 1.2])
    ax1.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Agreement metrics
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['Agreement\nRate (%)', 'Adjusted\nRand Index']
    values = [comparison['agreement'] * 100, comparison['ari']]
    bars = ax2.bar(metrics, values, color='darkgreen', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Cluster Agreement Metrics', fontsize=12, pad=10)
    ax2.grid(alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Crosstab heatmap
    ax3 = fig.add_subplot(gs[1, :])
    sns.heatmap(comparison['crosstab'], annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Sample Count'}, ax=ax3, linewidths=0.5)
    ax3.set_title('Cluster Correspondence: K-Means vs GMM', fontsize=12, pad=10)
    ax3.set_xlabel('GMM Clusters', fontsize=11)
    ax3.set_ylabel('K-Means Clusters', fontsize=11)

    plt.savefig(os.path.join(save_dir, "clustering_comparison_summary.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("- Clustering comparison summary plot saved")


def plot_parallel_coordinates(df, kmeans_labels, gmm_labels, feature_cols, save_dir, n_features=8):
    """Creează parallel coordinates plot pentru compararea clusterelor."""
    from pandas.plotting import parallel_coordinates

    # Selectează primele n_features pentru lizibilitate
    selected_features = feature_cols[:n_features]

    # K-Means
    df_kmeans = df[selected_features].copy()
    df_kmeans['Cluster'] = [f'K{i}' for i in kmeans_labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    parallel_coordinates(df_kmeans, 'Cluster', ax=ax1, alpha=0.3)
    ax1.set_title('K-Means Clusters - Parallel Coordinates', fontsize=13, pad=10)
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # GMM
    df_gmm = df[selected_features].copy()
    df_gmm['Cluster'] = [f'G{i}' for i in gmm_labels]

    parallel_coordinates(df_gmm, 'Cluster', ax=ax2, alpha=0.3)
    ax2.set_title('GMM Clusters - Parallel Coordinates', fontsize=13, pad=10)
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "parallel_coordinates.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("- Parallel coordinates plot saved")


def plot_radar_chart(comparison_df, save_dir, n_features=8):
    """Creează radar chart pentru compararea medie clusterelor."""
    from math import pi

    # Normalizează datele pentru vizualizare
    normalized_df = (comparison_df - comparison_df.min()) / (comparison_df.max() - comparison_df.min())

    # Selectează primele n_features
    selected_cols = normalized_df.columns[:n_features]
    categories = list(selected_cols)
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = plt.cm.tab10.colors

    for idx, cluster in enumerate(normalized_df.index):
        values = normalized_df.loc[cluster, selected_cols].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=cluster, color=colors[idx % 10])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % 10])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_title('Cluster Profiles - Radar Chart', size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cluster_radar_chart.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("- Radar chart saved")

def save_cluster_comparison_table(df, kmeans_labels, gmm_labels, feature_cols, save_dir):
    """Salvează tabel comparativ cu medii per cluster pentru ambele metode."""
    df_temp = df.copy()
    df_temp['KMeans_Cluster'] = kmeans_labels
    df_temp['GMM_Cluster'] = gmm_labels

    # Medii K-Means
    kmeans_means = df_temp.groupby('KMeans_Cluster')[feature_cols].mean()
    kmeans_means.index = [f'KMeans_C{i}' for i in kmeans_means.index]

    # Medii GMM
    gmm_means = df_temp.groupby('GMM_Cluster')[feature_cols].mean()
    gmm_means.index = [f'GMM_C{i}' for i in gmm_means.index]

    # Combină
    comparison_df = pd.concat([kmeans_means, gmm_means])

    # Salvează
    comparison_path = os.path.join(save_dir, "cluster_comparison_means.csv")
    comparison_df.to_csv(comparison_path)
    print(f"- Cluster comparison table saved to: {comparison_path}")

    return comparison_df