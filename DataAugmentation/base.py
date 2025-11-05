import pandas as pd
import os
import json
from EDA.V1.data_loading import load_data
from EDA.V1.visualization import visualize_data
from EDA.V1.model_training import evaluate_overfitting
from FirstProcessing.risk_calculation import scale_numeric_columns
from FirstProcessing.file_operations import auto_adjust_column_width


class BaseAugmentation:

    def __init__(self, target_column="Behavior_Risk_Level"):
        self.target_column = target_column
        self.metrics_history = []

    def load_dataset(self):
        return load_data()

    def prepare_data(self, df):
        excluded = ["Risk_Score", "Confidence", "Outlier", "Cluster", "Auto_Label"]

        # Drop excluded columns if they exist
        df_clean = df.drop(columns=[col for col in excluded if col in df.columns])

        X = df_clean.drop(columns=[self.target_column])
        y = df_clean[self.target_column]

        return X, y

    def scale_features(self, X, numeric_cols=None):
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        X_scaled = X.copy()
        X_scaled = scale_numeric_columns(X_scaled, numeric_cols)
        return X_scaled

    def save_results(self, X_aug, y_aug, metrics, save_dir=None):
        if save_dir is None:
            from EDA import select_save_directory
            save_dir = select_save_directory()
            if save_dir is None:
                return False

        # Combine features and target
        df_aug = pd.concat([X_aug, pd.Series(y_aug, name=self.target_column)], axis=1)

        # Save CSV (encoded only)
        csv_path = os.path.join(save_dir, "augmented_dataset_encoded.csv")
        df_aug.to_csv(csv_path, index=False)

        # Save Excel with encoded sheet only
        excel_path = os.path.join(save_dir, "augmented_dataset_encoded.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_aug.to_excel(writer, index=False, sheet_name='Encoded_Data')
            auto_adjust_column_width(writer, 'Encoded_Data')

        # Format metrics
        if isinstance(metrics, list):
            formatted_metrics = {"Augmentation": {
                "augmentation_history": metrics,
                "f1_score": 0.0,
                "classification_report": {},
                "roc_auc": None,
                "cv_f1_mean": 0.0,
                "cv_f1_std": 0.0
            }}
        else:
            formatted_metrics = metrics

        # Save metrics JSON
        metrics_path = os.path.join(save_dir, "augmentation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(formatted_metrics, f, indent=2)

        print(f"\nSaved encoded dataset (CSV, XLSX) and metrics to: {save_dir}")
        return True

    def visualize_results(self, df_aug, models_results=None):
        return visualize_data(df_aug, models_results=models_results)

    def evaluate_models(self, models, X_train, X_test, y_train, y_test):
        return evaluate_overfitting(models, X_train, X_test, y_train, y_test)
