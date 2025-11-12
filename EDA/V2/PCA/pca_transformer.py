import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import TARGET


def fit_pca(df, variance_threshold=0.80):

    numeric_cols = df.select_dtypes(include=np.number).columns
    feature_cols = [c for c in numeric_cols if c != TARGET]
    X = df[feature_cols].dropna()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit full PCA
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Determine components
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Fit final PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    return {
        'scaler': scaler,
        'pca_model': pca,
        'pca_full': pca_full,
        'n_components': n_components,
        'feature_cols': feature_cols,
        'X_scaled': X_scaled,
        'cumulative_variance': cumulative_variance
    }


def transform_pca(X_scaled, pca_model):
    return pca_model.transform(X_scaled)


def get_loadings(pca_model, feature_cols):
    return pd.DataFrame(
        pca_model.components_.T,
        columns=[f'PC{i + 1}' for i in range(pca_model.n_components_)],
        index=feature_cols
    )
