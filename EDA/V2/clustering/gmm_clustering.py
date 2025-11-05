import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def run_gmm(X, n_components):
    """
    Run Gaussian Mixture Model clustering.

    Returns:
        dict with GMM results
    """
    gmm = GaussianMixture(n_components=n_components,
                          random_state=42,
                          covariance_type='full')
    gmm.fit(X)

    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)
    silhouette = silhouette_score(X, labels)

    results = {
        'model': gmm,
        'labels': labels,
        'probabilities': probabilities,
        'silhouette': silhouette,
        'bic': gmm.bic(X),
        'aic': gmm.aic(X)
    }

    print(f"GMM (K={n_components}):")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  BIC: {results['bic']:.2f}")
    print(f"  AIC: {results['aic']:.2f}")

    return results


def save_optimal_gmm_dataset(df, labels, probabilities, save_dir):
    """Save dataset with optimal GMM cluster labels and probabilities."""
    import os

    df_with_clusters = df.copy()
    df_with_clusters['cluster_gmm'] = labels

    # Add max probability as confidence score
    df_with_clusters['cluster_confidence'] = probabilities.max(axis=1)

    output_path = os.path.join(save_dir, "dataset_with_gmm_clusters.csv")
    df_with_clusters.to_csv(output_path, index=False)

    print(f"- GMM dataset saved: {output_path}")
    return df_with_clusters
