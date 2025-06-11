import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import json
import pandas as pd
import numpy as np
import gc
import torch
import time
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp
from FirstProcessing.risk_calculation import calculate_risk_advanced
from EDA.data_loading import load_data
from EDA.file_operations import select_save_directory
from FirstProcessing.file_operations import auto_adjust_column_width
import warnings
warnings.filterwarnings("ignore")

class WCGANAugmentation:
    def __init__(self, target_column="Behavior_Risk_Level", step_fraction=0.25, max_size=2000,
                 random_state=42, min_confidence=0.8):
        self.target_column = target_column
        self.step_fraction = step_fraction
        self.max_size = max_size
        self.random_state = random_state
        self.min_confidence = min_confidence
        self.device = torch.device("cpu")
        self.history = []

    def prepare_data(self, df):
        excluded = ['Risk_Score', 'Confidence', 'Outlier', 'Cluster', 'Auto_Label']
        df = df.drop(columns=[col for col in excluded if col in df.columns], errors='ignore')
        return df.drop(columns=[self.target_column]), df[self.target_column]

    def build_metadata(self, df):
        from sdv.metadata import Metadata

        excluded_columns = {
            'Risk_Score', 'Confidence', 'Outlier', 'Cluster', 'Auto_Label'
        }

        print(" Generating metadata with manual column type assignment...")

        columns_metadata = {}

        for col in df.columns:
            if col in excluded_columns:
                print(f" Skipping excluded column: {col}")
                continue

            dtype = df[col].dtype
            unique_values = df[col].dropna().unique()

            if dtype == 'float64':
                sdtype = 'numerical'
            elif dtype == 'int64':
                if set(unique_values).issubset({0, 1}):
                    sdtype = 'boolean'
                elif len(unique_values) < 20:
                    sdtype = 'categorical'
                else:
                    sdtype = 'numerical'
            else:
                sdtype = 'categorical'

            columns_metadata[col] = {'sdtype': sdtype}
            print(f" {col}: {dtype} → SDV type: {sdtype}")

        metadata_dict = {'columns': columns_metadata}
        metadata = Metadata.load_from_dict(metadata_dict)

        # Return both the dictionary and the Metadata object
        return metadata_dict, metadata

    def generate_step(self, df_train, n_samples):
        metadata_dict, metadata = self.build_metadata(df_train)
        # Remove columns not in metadata
        included_cols = list(metadata_dict['columns'].keys())
        df_train = df_train[included_cols].copy()

        # Process boolean columns
        columns_metadata = metadata_dict["columns"]
        bool_cols = [col for col, meta in columns_metadata.items() if meta.get("sdtype") == "boolean"]
        df_train[bool_cols] = df_train[bool_cols].astype(bool)

        # Use the metadata object with the synthesizer
        model = CTGANSynthesizer(metadata, epochs=1000, batch_size=150, cuda=False)
        model.fit(df_train)

        # Use basic sampling without conditions
        print("Generating samples without conditions (not supported in SDV 1.22.1)...")
        samples = model.sample(n_samples)

        # Manual post-processing to balance classes
        if self.target_column in samples.columns:
            # Check current distribution
            current_dist = samples[self.target_column].value_counts()
            print(f"Generated distribution: {current_dist.to_dict()}")

            # If very unbalanced, try to adjust
            if min(current_dist) < n_samples * 0.3:
                print("Attempting to balance classes in generated data...")
                # Generate more samples and select a balanced subset
                extra_samples = model.sample(n_samples * 2)
                all_samples = pd.concat([samples, extra_samples])

                # Select balanced samples from each class
                balanced = []
                target_per_class = n_samples // 2
                for class_val in [0, 1]:
                    class_samples = all_samples[all_samples[self.target_column] == class_val]
                    if len(class_samples) >= target_per_class:
                        balanced.append(class_samples.sample(target_per_class))
                    else:
                        balanced.append(class_samples)

                samples = pd.concat(balanced)
                print(f"Balanced distribution: {samples[self.target_column].value_counts().to_dict()}")

        return samples

    def relabel_confident(self, original_df, generated_df):
        X_orig, y_orig = self.prepare_data(original_df)
        X_gen, y_gen = self.prepare_data(generated_df)

        clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf.fit(X_orig, y_orig)

        proba = clf.predict_proba(X_gen)
        confidence = np.max(proba, axis=1)
        delta = np.abs(proba[:, 1] - proba[:, 0])

        # Filtrare de outlieri pe baza încrederii și polarizării predicției
        mask = (confidence >= self.min_confidence) & (delta > 0.5)
        print(f"\n Filtering: Kept {np.sum(mask)} / {len(generated_df)} samples (confident & clear)")

        # Aplică mască și relabel doar pe mostrele bune
        X_gen_filtered = X_gen[mask]
        y_gen_filtered = y_gen[mask]
        y_pred_filtered = clf.predict(X_gen_filtered)

        changed = np.sum(y_pred_filtered != y_gen_filtered)
        y_gen_filtered = y_gen_filtered.copy()
        y_gen_filtered[:] = y_pred_filtered  # suprascrie cu predicții

        # Returnează doar mostrele filtrate
        generated_df_filtered = generated_df.iloc[mask].copy()
        generated_df_filtered[self.target_column] = y_gen_filtered.astype(float)


        print(f"Relabeled {np.sum(mask)} samples, with {changed} changes.")
        return generated_df_filtered

    def filter_consistency(self, df):
        if "Save_Money_Yes" not in df.columns or "Save_Money_No" not in df.columns:
            return df  # nimic de verificat

        goal_cols = [col for col in df.columns if col.startswith("Savings_Goal_")]
        obs_cols = [col for col in df.columns if col.startswith("Savings_Obstacle_")]

        mask_goal_valid = ~((df["Save_Money_No"] == 1) & (df[goal_cols].any(axis=1)))
        mask_obs_valid = ~((df["Save_Money_Yes"] == 1) & (df[obs_cols].any(axis=1)))

        consistent_df = df[mask_goal_valid & mask_obs_valid].copy()
        print(f" Consistency filter: Kept {len(consistent_df)} / {len(df)} samples")

        return consistent_df

    def validate_new_samples(self, original_df, augmented_df):
        X_orig, y_orig = self.prepare_data(original_df)
        X_new, y_new = self.prepare_data(augmented_df)

        clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf.fit(X_orig, y_orig)
        y_pred = clf.predict(X_new)

        y_new = y_new.astype(float)
        y_pred = y_pred.astype(float)

        kappa = cohen_kappa_score(y_new, y_pred)
        f1 = f1_score(y_new, y_pred, average='weighted')

        # Select only numeric columns (float/int) and compute means as NumPy arrays
        numeric_cols = X_orig.select_dtypes(include=[np.number]).columns
        mean_orig = X_orig[numeric_cols].mean().values.astype(np.float64)
        mean_new = X_new[numeric_cols].mean().values.astype(np.float64)
        cos_sim = 1 - cosine(mean_orig, mean_new)

        # KS test for distribution shift
        ks_results = {
            col: ks_2samp(X_orig[col], X_new[col])[0]
            for col in numeric_cols
        }

        return {
            'cohen_kappa': kappa,
            'f1_score': f1,
            'cosine_similarity': cos_sim,
            'ks_distance': ks_results
        }

    def save_results(self, df_aug, history, save_dir):
        bool_cols = df_aug.select_dtypes(include=["bool"]).columns
        df_aug[bool_cols] = df_aug[bool_cols].astype(int)
        df_aug.to_csv(os.path.join(save_dir, "augmented_dataset_encoded.csv"), index=False)

        excel_path = os.path.join(save_dir, "augmented_dataset_encoded.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_aug.to_excel(writer, index=False, sheet_name='Encoded_Data')
            auto_adjust_column_width(writer, 'Encoded_Data')

        with open(os.path.join(save_dir, "augmentation_metrics.json"), 'w') as f:
            json.dump({"augmentation_history": history}, f, indent=2)

        print(f"\nSaved to {save_dir}")

    def augment_incrementally(self, df):
        df_current = df.copy()
        iteration = 0
        start_time = time.time()

        iteration_durations = []
        train_sizes = []

        while len(df_current) < self.max_size:
            iteration += 1
            iter_start = time.time()
            to_add = min(max(1, int(len(df_current) * self.step_fraction)), self.max_size - len(df_current))

            print(f"\nIteration {iteration}: Generating {to_add} samples...")

            generated = self.generate_step(df_current, to_add)

            if self.target_column not in generated.columns:
                print("Generated samples missing target column. Skipping iteration.")
                continue

            generated = self.relabel_confident(df_current, generated)
            generated = self.filter_consistency(generated)

            metrics = self.validate_new_samples(df_current, generated)
            self.history.append({
                'iteration': iteration,
                'generated': len(generated),
                'new_total': len(df_current) + len(generated),
                'metrics': metrics
            })

            df_current = pd.concat([df_current, generated], ignore_index=True)
            gc.collect()

            elapsed_iter = time.time() - iter_start
            total_elapsed = time.time() - start_time

            iteration_durations.append(elapsed_iter)
            train_sizes.append(len(df_current))

            # ETA realistă pe bază de regresie timp / mărime
            if len(train_sizes) >= 3:
                X = np.array(train_sizes).reshape(-1, 1)
                y = np.array(iteration_durations)
                reg = LinearRegression().fit(X, y)

                remaining_instances = self.max_size - len(df_current)
                predicted_time_next = reg.predict([[len(df_current) + to_add]])[0]
                estimated_remaining = predicted_time_next * (remaining_instances / to_add)
            else:
                avg_time = np.mean(iteration_durations)
                estimated_remaining = avg_time * ((self.max_size - len(df_current)) / to_add)

            print(
                f"Validation - F1: {metrics['f1_score']:.3f}, Kappa: {metrics['cohen_kappa']:.3f}, CosSim: {metrics['cosine_similarity']:.3f}")
            print(
                f"Time: {elapsed_iter:.2f}s | Total: {total_elapsed:.1f}s | ETA: {estimated_remaining:.1f}s | Progress: {len(df_current) / self.max_size * 100:.1f}%")

        X_orig, y_orig = self.prepare_data(df)
        X_full, y_full = self.prepare_data(df_current)
        df_current[self.target_column] = y_full

        return df_current, self.history


if __name__ == "__main__":
    print("Using device: CPU (forced)")
    augmenter = WCGANAugmentation(
        target_column="Behavior_Risk_Level",
        step_fraction=0.25,
        max_size=10000,
        random_state=42,
        min_confidence=0.8
    )

    df = load_data()
    if df is not None:
        df_aug, history = augmenter.augment_incrementally(df)

        print("\nAugmentation Complete!")
        print(f"Original distribution: {df[augmenter.target_column].value_counts().to_dict()}")
        print(f"Augmented distribution: {df_aug[augmenter.target_column].value_counts().to_dict()}")
        print(f"Total samples: {len(df_aug)}")

        save_dir = select_save_directory()
        if save_dir:
            augmenter.save_results(df_aug, history, save_dir)
