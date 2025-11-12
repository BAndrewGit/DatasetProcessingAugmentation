import numpy as np
import pandas as pd
import os
from sklearn.metrics import adjusted_rand_score

#Compare K-Means and GMM clustering results
def compare_clustering_methods(kmeans_labels, gmm_labels):
    ari = adjusted_rand_score(kmeans_labels, gmm_labels)
    crosstab = pd.crosstab(
        pd.Series(kmeans_labels, name='K-Means'),
        pd.Series(gmm_labels, name='GMM')
    )
    agreement = np.mean(kmeans_labels == gmm_labels)

    results = {
        'ari': ari,
        'crosstab': crosstab,
        'agreement': agreement
    }

    print("\n=== K-Means vs GMM Comparison ===")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Agreement Rate: {agreement * 100:.2f}%")

    return results

# Analyze Risk_Score distribution across clusters for both methods
def compare_risk_score_distribution(df_kmeans, df_gmm, target='Risk_Score'):
    kmeans_stats = df_kmeans.groupby('cluster_kmeans')[target].agg([
        'count', 'mean', 'std', 'min', 'median', 'max'
    ]).round(3)

    gmm_stats = df_gmm.groupby('cluster_gmm')[target].agg([
        'count', 'mean', 'std', 'min', 'median', 'max'
    ]).round(3)

    print("\n=== Risk_Score Distribution ===")
    print("\nK-Means:")
    print(kmeans_stats)
    print("\nGMM:")
    print(gmm_stats)

    return {
        'kmeans_stats': kmeans_stats,
        'gmm_stats': gmm_stats
    }


def save_clustering_summary(kmeans_results, gmm_results, comparison,
                            risk_stats, k_optimal, save_dir):

    summary_data = {
        'Metric': [
            'Optimal K',
            'K-Means Silhouette',
            'GMM Silhouette',
            'Silhouette Difference',
            'Adjusted Rand Index',
            'Agreement Rate (%)',
            'GMM BIC',
            'GMM AIC'
        ],
        'Value': [
            k_optimal,
            kmeans_results[k_optimal]['silhouette'],
            gmm_results['silhouette'],
            abs(kmeans_results[k_optimal]['silhouette'] - gmm_results['silhouette']),
            comparison['ari'],
            comparison['agreement'] * 100,
            gmm_results['bic'],
            gmm_results['aic']
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_dir, "clustering_summary.csv"),
                      index=False)

    # Combined Risk Score analysis
    with pd.ExcelWriter(os.path.join(save_dir, "risk_score_analysis.xlsx")) as writer:
        risk_stats['kmeans_stats'].to_excel(writer, sheet_name='K-Means')
        risk_stats['gmm_stats'].to_excel(writer, sheet_name='GMM')

        # Comparison sheet
        comparison_df = pd.DataFrame({
            'Cluster': risk_stats['kmeans_stats'].index,
            'KMeans_Mean': risk_stats['kmeans_stats']['mean'].values,
            'GMM_Mean': risk_stats['gmm_stats']['mean'].values,
            'Difference': (risk_stats['kmeans_stats']['mean'] -
                           risk_stats['gmm_stats']['mean']).values
        })
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)

    print(f"\n- Clustering summary saved to: {save_dir}")
