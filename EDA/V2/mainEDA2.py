import os
import pandas as pd
from utils import select_output_directory, create_plot_directories
from data_loader import load_and_prepare_data, build_group_map
from plot_generator import (generate_univariate_plots,
                            generate_bivariate_plots,
                            generate_target_plots)
from PCA import (fit_pca, transform_pca, get_loadings,
                 plot_scree, plot_loadings_heatmap, save_pca_results)
from config import PCA_VARIANCE_THRESHOLD


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

    # Fit PCA
    pca_results = fit_pca(df, variance_threshold=PCA_VARIANCE_THRESHOLD)
    print(f"Number of components for {PCA_VARIANCE_THRESHOLD*100:.0f}% variance: "
          f"{pca_results['n_components']}")

    # Transform data
    X_pca = transform_pca(pca_results['X_scaled'], pca_results['pca_model'])
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(pca_results['n_components'])]
    )

    # Get loadings
    loadings_df = get_loadings(pca_results['pca_model'], pca_results['feature_cols'])

    # Generate visualizations
    plot_scree(pca_results['pca_full'],
               pca_results['cumulative_variance'],
               pca_results['n_components'],
               PCA_VARIANCE_THRESHOLD,
               pca_dir)

    plot_loadings_heatmap(loadings_df, pca_dir)

    # Save results
    save_pca_results(loadings_df, pca_df, pca_dir)

    # Print summary
    print("\nVariance explained by each component:")
    for i, var in enumerate(pca_results['pca_model'].explained_variance_ratio_, 1):
        print(f"  PC{i}: {var:.4f} ({var*100:.2f}%)")

    return pca_results, pca_df


def main():
    # Setup
    plots_dir = select_output_directory()
    create_plot_directories(plots_dir)

    # Load data
    df = load_and_prepare_data()
    group_map = build_group_map(df)

    # Run analyses
    run_eda_plots(df, group_map, plots_dir)
    pca_results, pca_df = run_pca_analysis(df, plots_dir)

    print(f"\n=== Analysis Complete ===")
    print(f"All results saved to: {plots_dir}")
    print(f"PCA components: {pca_results['n_components']}")


if __name__ == "__main__":
    main()
