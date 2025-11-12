import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Run K-Means clustering over a range of K values
def run_kmeans_range(X, k_range=(2, 11)):

    results = {}

    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        inertia = kmeans.inertia_

        results[k] = {
            'model': kmeans,
            'labels': labels,
            'silhouette': silhouette,
            'inertia': inertia,
            'centers': kmeans.cluster_centers_
        }

        print(f"K={k}: Silhouette={silhouette:.4f}, Inertia={inertia:.2f}")

    return results


def find_optimal_k(results, method='silhouette'):

    if method == 'silhouette':
        k_optimal = max(results.keys(),
                        key=lambda k: results[k]['silhouette'])
    else:  # elbow method
        inertias = [results[k]['inertia'] for k in sorted(results.keys())]
        diffs = np.diff(inertias)
        k_optimal = sorted(results.keys())[np.argmin(diffs) + 1]

    print(f"\nOptimal K ({method}): {k_optimal}")
    print(f"Silhouette Score: {results[k_optimal]['silhouette']:.4f}")

    return k_optimal


def save_optimal_kmeans_dataset(df, labels, save_dir):

    df_with_clusters = df.copy()
    df_with_clusters['cluster_kmeans'] = labels

    output_path = os.path.join(save_dir, "dataset_with_kmeans_clusters.csv")
    df_with_clusters.to_csv(output_path, index=False)

    print(f"- K-Means dataset saved: {output_path}")
    return df_with_clusters
