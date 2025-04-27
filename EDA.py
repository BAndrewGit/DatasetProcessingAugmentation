import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    ConfusionMatrixDisplay,
    silhouette_score,
    roc_curve,
    auc
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import json
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
pd.set_option('future.no_silent_downcasting', True)

CONFIG = {
    'required_columns': [
        'Age', 'Essential_Needs_Percentage', 'Expense_Distribution_Entertainment',
        'Debt_Level', 'Save_Money_Yes', 'Behavior_Risk_Level'
    ],
    'risk_weights': [0.3, 0.25, 0.35, 0.1],
    'test_size': 0.25,
    'kmeans_clusters': 3,
    'min_samples_cluster': 10
}

def load_data():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return pd.read_csv(file_path) if file_path else None

def preprocess_encoded_data(df):
    for col in CONFIG['required_columns']:
        if col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
    return df

def train_models(df):
    try:
        drop_cols = ['Behavior_Risk_Level', 'Risk_Score', 'Cluster', 'Outlier', 'Auto_Label', 'Confidence']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df['Behavior_Risk_Level']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=CONFIG['test_size'],
            stratify=y,
            random_state=42
        )

        models = {
            'LogisticRegression': LogisticRegression(
                class_weight='balanced',
                solver='newton-cg',
                max_iter=2000,
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=150,
                class_weight='balanced',
                random_state=42
            )
        }

        fallback_flag = False
        print("\nTraining SVM...")
        with warnings.catch_warnings(record=True) as w:
            svm_model = SVC(
                kernel='linear',
                probability=True,
                class_weight='balanced',
                random_state=42,
                max_iter=500
            )
            svm_model.fit(X_train, y_train)

            if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                print("⚠️ SVM did not converge! Increasing max_iter to 2000...")
                svm_model = SVC(
                    kernel='linear',
                    probability=True,
                    class_weight='balanced',
                    random_state=42,
                    max_iter=2000
                )
                svm_model.fit(X_train, y_train)

                with warnings.catch_warnings(record=True) as w2:
                    svm_model.fit(X_train, y_train)
                    if any(issubclass(warn.category, ConvergenceWarning) for warn in w2):
                        print("⚠️ SVM still not converging, switching to SGDClassifier with calibration...")
                        from sklearn.linear_model import SGDClassifier
                        from sklearn.calibration import CalibratedClassifierCV
                        base_sgd = SGDClassifier(
                            loss='hinge',
                            max_iter=10000,
                            class_weight='balanced',
                            random_state=42,
                            tol=1e-3
                        )
                        svm_model = CalibratedClassifierCV(estimator=base_sgd, cv=5)
                        svm_model.fit(X_train, y_train)
                        fallback_flag = True

        if fallback_flag:
            models['SGDClassifier'] = svm_model
        else:
            models['SVM'] = svm_model

        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            report = classification_report(y_test, y_pred, target_names=['Beneficially', 'Risky'], output_dict=True)
            auc_score = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            f1 = f1_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')

            results[name] = {
                'classification_report': report,
                'roc_auc': auc_score,
                'f1_score': f1,
                'cv_f1_mean': np.mean(cv_scores),
                'cv_f1_std': np.std(cv_scores),
                'model': model
            }

            disp = ConfusionMatrixDisplay(
                confusion_matrix(y_test, y_pred, normalize='true'),
                display_labels=['Beneficially', 'Risky']
            )
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.show()

            if y_proba is not None:
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {name}')
                plt.legend(loc="lower right")
                plt.show()

        return models, results, X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error training models: {str(e)}")
        return None, None, None, None, None, None

def evaluate_overfitting(models, X_train, X_test, y_train, y_test):
    overfit_report = {}
    for name, model in models.items():
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        overfit_ratio = train_acc - test_acc
        overfit_report[name] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "overfit_ratio": overfit_ratio
        }
        print(f"{name} - Train accuracy: {train_acc:.3f}, Test accuracy: {test_acc:.3f}, Overfit ratio: {overfit_ratio:.3f}")

    return overfit_report

def select_save_directory():
    Tk().withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected if folder_selected else None

def save_plot(fig, save_dir, filename):
    if save_dir:
        path = os.path.join(save_dir, filename)
        fig.savefig(path)
        plt.close(fig)

def visualize_data(df, models_results=None):
    try:
        save_dir = select_save_directory()
        df_viz = df.copy()
        df_viz['Behavior_Risk_Label'] = df_viz['Behavior_Risk_Level'] \
            .map({0: 'Beneficially', 1: 'Risky'})

        # 1. Risk Level Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            x='Behavior_Risk_Label', data=df_viz,
            hue='Behavior_Risk_Label', legend=False,
            palette='Set2', ax=ax
        )
        ax.set_title('Risk Level Distribution')
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Number of Cases')
        plt.tight_layout()
        save_plot(fig, save_dir, 'risk_level_distribution.png')

        # 2. Distribution of Risk Factors
        risk_factors = [
            'Essential_Needs_Percentage',
            'Expense_Distribution_Entertainment',
            'Debt_Level',
            'Save_Money_Yes'
        ]
        scaler = StandardScaler()
        df_viz[risk_factors] = scaler.fit_transform(df_viz[risk_factors])

        df_risk = df_viz[risk_factors + ['Behavior_Risk_Label']]
        df_melted = df_risk.melt(
            id_vars='Behavior_Risk_Label',
            var_name='Factor',
            value_name='Value'
        )
        # replace underscores with spaces for better readability
        df_melted['Factor'] = df_melted['Factor'].str.replace('_', ' ')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            x='Factor', y='Value',
            hue='Behavior_Risk_Label',
            data=df_melted,
            palette='Set3',
            dodge=True,
            ax=ax
        )
        ax.set_title('Distribution of Risk Factors')
        ax.set_xlabel('Factor')
        ax.set_ylabel('Standardized Value')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.legend(
            title='Risk Level',
            loc='upper right',
            fontsize=9,
            title_fontsize=10
        )
        plt.tight_layout()
        save_plot(fig, save_dir, 'distribution_risk_factors.png')

        # 3. Top Features Correlation Heatmap
        numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
        corr = df_viz[numeric_cols].corr()
        # pick top 20 by absolute correlation
        top_pairs = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
        top_features = list({i for i, j in top_pairs.head(20).index})

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            df_viz[top_features].corr(),
            cmap='coolwarm', annot=False, ax=ax
        )
        ax.set_title('Top Features Correlation Heatmap')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        save_plot(fig, save_dir, 'top_features_correlation_heatmap.png')

        # 4. Learning Curve for Logistic Regression
        X_lr = df_viz.drop(columns=['Behavior_Risk_Level', 'Behavior_Risk_Label'])
        y_lr = df_viz['Behavior_Risk_Level']
        train_sizes, train_scores, test_scores = learning_curve(
            LogisticRegression(
                max_iter=1000,
                solver='newton-cg',
                random_state=42
            ),
            X_lr, y_lr,
            cv=5, scoring='f1',
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Train F1 score')
        ax.plot(train_sizes, np.mean(test_scores, axis=1), label='CV F1 score')
        ax.set_title('Learning Curve for Logistic Regression')
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel('F1 Score')
        ax.legend(loc='best')
        plt.tight_layout()
        save_plot(fig, save_dir, 'learning_curve_logistic.png')

        # 5. Confusion Matrices & ROC Curves
        if models_results:
            for model_name, (model, X_test, y_test) in models_results.items():
                # confusion matrix
                fig, ax = plt.subplots(figsize=(7, 6))
                ConfusionMatrixDisplay.from_estimator(
                    model, X_test, y_test,
                    normalize='true', cmap='Blues', ax=ax
                )
                ax.set_title(f'Confusion Matrix – {model_name}', pad=20)
                plt.tight_layout()
                save_plot(fig, save_dir, f'confusion_matrix_{model_name}.png')

                # ROC curve
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)

                    fig, ax = plt.subplots(figsize=(7, 6))
                    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_title(f'ROC Curve – {model_name}', pad=20)
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.legend(loc='lower right')
                    plt.tight_layout()
                    save_plot(fig, save_dir, f'roc_curve_{model_name}.png')

        return df_viz

    except Exception as e:
        print(f"Error in visualizations: {e}")
        return None

def save_metrics(results, overfit_report, feature_names):
    try:
        metrics = {}
        for name, res in results.items():
            metrics[name] = {
                'classification_report': res['classification_report'],
                'roc_auc': res['roc_auc'],
                'f1_score': res['f1_score'],
                'cv_f1_mean': res['cv_f1_mean'],
                'cv_f1_std': res['cv_f1_std'],
                'train_accuracy': overfit_report[name]['train_accuracy'],
                'test_accuracy': overfit_report[name]['test_accuracy'],
                'overfit_ratio': overfit_report[name]['overfit_ratio']
            }
            if name == 'LogisticRegression':
                coef_dict = {
                    feature: coef for feature, coef in zip(feature_names, res['model'].coef_[0])
                }
                sorted_coefs = dict(sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True))
                metrics[name]['coefficients'] = sorted_coefs

        with open('dataset_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print("Extended metrics have been saved to 'dataset_metrics.json'.")

    except Exception as e:
        print(f"Error saving metrics: {str(e)}")

def main():
    df = load_data()
    if df is None:
        return

    df = preprocess_encoded_data(df)
    if df is None:
        return

    models, results, X_train, X_test, y_train, y_test = train_models(df)
    if models is None:
        return

    # Adăugăm dict-ul de modele pentru Confusion și ROC
    models_results = {
        name: (model, X_test, y_test) for name, model in models.items()
    }

    df_viz = visualize_data(df, models_results=models_results)

    overfit_report = evaluate_overfitting(models, X_train, X_test, y_train, y_test)
    save_metrics(results, overfit_report, feature_names=X_train.columns)

if __name__ == "__main__":
    main()
