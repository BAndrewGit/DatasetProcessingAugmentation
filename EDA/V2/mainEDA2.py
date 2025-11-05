import os
import pandas as pd
from utils import select_output_directory, create_plot_directories
from data_loader import load_and_prepare_data, build_group_map
from plot_generator import (generate_univariate_plots,
                            generate_bivariate_plots,
                            generate_target_plots)
from PCA import (fit_pca, transform_pca, get_loadings,
                 plot_scree, plot_loadings_heatmap, save_pca_results)
from clustering import (run_kmeans_range, find_optimal_k, save_optimal_kmeans_dataset,
                       run_gmm, save_optimal_gmm_dataset,
                       compare_clustering_methods, compare_risk_score_distribution,
                       save_clustering_summary,
                       plot_silhouette_scores, plot_pca_clusters,
                       plot_risk_score_by_cluster, plot_cluster_comparison_summary)
from config import PCA_VARIANCE_THRESHOLD, CLUSTERING_K_RANGE


def run_eda_plots(df, group_map, plots_dir):
    """Run all EDA visualizations."""
    print("\n=== Starting EDA Visualizations ===")
    generate_univariate_plots(df, group_map, plots_dir)
    generate_bivariate_plots(df, group_map, plots_dir)
    generate_target_plots(df, plots_dir)


def run_pca_analysis(df, plots_dir):
    """Run PCA analysis with visualizations."""
    print("\n=== Starting PCA Analysis ===")
    pca_dir = os.path.join(plots_dir, "pca")
    os.makedirs(pca_dir, exist_ok=True)

    pca_results = fit_pca(df, variance_threshold=PCA_VARIANCE_THRESHOLD)
    print(f"Number of components: {pca_results['n_components']}")

    X_pca = transform_pca(pca_results['X_scaled'], pca_results['pca_model'])
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(pca_results['n_components'])]
    )

    loadings_df = get_loadings(pca_results['pca_model'], pca_results['feature_cols'])

    plot_scree(pca_results['pca_full'],
               pca_results['cumulative_variance'],
               pca_results['n_components'],
               PCA_VARIANCE_THRESHOLD,
               pca_dir)
    plot_loadings_heatmap(loadings_df, pca_dir)
    save_pca_results(loadings_df, pca_df, pca_dir)

    return pca_results, X_pca, pca_df


def run_clustering_analysis(df, X_pca, plots_dir):
    """Run clustering analysis with K-Means and GMM."""
    print("\n=== Starting Clustering Analysis ===")
    cluster_dir = os.path.join(plots_dir, "clustering")
    os.makedirs(cluster_dir, exist_ok=True)

    # K-Means
    print("\n--- K-Means Clustering ---")
    kmeans_results = run_kmeans_range(X_pca, k_range=CLUSTERING_K_RANGE)
    k_optimal = find_optimal_k(kmeans_results, method='silhouette')
    kmeans_labels = kmeans_results[k_optimal]['labels']
    df_kmeans = save_optimal_kmeans_dataset(df, kmeans_labels, cluster_dir)

    # GMM
    print("\n--- GMM Clustering ---")
    gmm_results = run_gmm(X_pca, n_components=k_optimal)
    df_gmm = save_optimal_gmm_dataset(df,
                                     gmm_results['labels'],
                                     gmm_results['probabilities'],
                                     cluster_dir)

    # Comparison
    print("\n--- Comparing Methods ---")
    comparison = compare_clustering_methods(kmeans_labels, gmm_results['labels'])
    risk_stats = compare_risk_score_distribution(df_kmeans, df_gmm)
    save_clustering_summary(kmeans_results, gmm_results, comparison,
                           risk_stats, k_optimal, cluster_dir)

    # Visualizations
    plot_silhouette_scores(kmeans_results, cluster_dir)
    plot_pca_clusters(X_pca, kmeans_labels, gmm_results['labels'], cluster_dir)
    plot_risk_score_by_cluster(df_kmeans, df_gmm, cluster_dir)
    plot_cluster_comparison_summary(comparison,
                                   kmeans_results[k_optimal]['silhouette'],
                                   gmm_results['silhouette'],
                                   cluster_dir)

    print(f"\nK-Means Silhouette: {kmeans_results[k_optimal]['silhouette']:.4f}")
    print(f"GMM Silhouette: {gmm_results['silhouette']:.4f}")

    return kmeans_results, gmm_results, k_optimal


def main():
    plots_dir = select_output_directory()
    create_plot_directories(plots_dir)

    df = load_and_prepare_data()
    group_map = build_group_map(df)

    run_eda_plots(df, group_map, plots_dir)
    pca_results, X_pca, pca_df = run_pca_analysis(df, plots_dir)
    kmeans_results, gmm_results, k_optimal = run_clustering_analysis(
        df, X_pca, plots_dir
    )

    print(f"\n=== Analysis Complete ===")
    print(f"All results saved to: {plots_dir}")
    print(f"PCA components: {pca_results['n_components']}")
    print(f"Optimal clusters: {k_optimal}")


if __name__ == "__main__":
    main()
