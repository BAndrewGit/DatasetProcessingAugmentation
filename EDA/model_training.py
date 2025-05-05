import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    roc_curve,
    auc, ConfusionMatrixDisplay
)
from FirstProcessing.risk_calculation import scale_numeric_columns

# Config for split and clustering
CONFIG = {
    'test_size': 0.25,
    'kmeans_clusters': 3,
    'min_samples_cluster': 10
}

# Train multiple classifiers and collect performance metrics
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

        # Define initial models
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

        # Register SVM or fallback SGD
        if fallback_flag:
            models['SGDClassifier'] = svm_model
        else:
            models['SVM'] = svm_model

        results = {}

        # Train and evaluate each model
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

            # Confusion matrix
            disp = ConfusionMatrixDisplay(
                confusion_matrix(y_test, y_pred, normalize='true'),
                display_labels=['Beneficially', 'Risky']
            )
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.show()

            # ROC curve
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

# Compare train/test accuracy for overfitting report
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