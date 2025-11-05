import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)
from FirstProcessing.risk_calculation import scale_numeric_columns

ARTIFICIAL_COLUMNS = ["Auto_Label", "Cluster", "Confidence", "Outlier", "Risk_Score"]

def drop_artificial_features(df):
    return df.drop(columns=[c for c in ARTIFICIAL_COLUMNS if c in df.columns])


def train_models(X_train, X_test, y_train, y_test):
    try:
        # XGB weight
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

        models = {
            'LogisticRegression': LogisticRegression(
                class_weight='balanced', solver='newton-cg',
                max_iter=2000, random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=150, class_weight='balanced', random_state=42
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, eval_metric='logloss',
                scale_pos_weight=scale_pos, random_state=42
            )
        }

        # SVM → optional SGD fallback
        fallback_flag = False
        print("\nTraining SVM...")
        with warnings.catch_warnings(record=True) as w:
            svm = SVC(
                kernel='linear', probability=True,
                class_weight='balanced', random_state=42,
                max_iter=500
            )
            svm.fit(X_train, y_train)
            if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                svm = SVC(
                    kernel='linear', probability=True,
                    class_weight='balanced', random_state=42,
                    max_iter=2000
                )
                svm.fit(X_train, y_train)
                with warnings.catch_warnings(record=True) as w2:
                    svm.fit(X_train, y_train)
                    if any(issubclass(warn.category, ConvergenceWarning) for warn in w2):
                        base = SGDClassifier(
                            loss='hinge', max_iter=10000,
                            class_weight='balanced', random_state=42,
                            tol=1e-3
                        )
                        svm = CalibratedClassifierCV(estimator=base, cv=5)
                        svm.fit(X_train, y_train)
                        fallback_flag = True

        if fallback_flag:
            models['SGDClassifier'] = svm
        else:
            models['SVM'] = svm

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            # fit on full train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # manual CV F1
            cv_scores = []
            for tr, val in skf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[tr], X_train.iloc[val]
                y_tr, y_val = y_train.iloc[tr], y_train.iloc[val]
                if y_tr.nunique() < 2:
                    continue
                model.fit(X_tr, y_tr)
                cv_scores.append(f1_score(y_val, model.predict(X_val)))

            results[name] = {
                'classification_report': classification_report(
                    y_test, y_pred, target_names=['Beneficially','Risky'], output_dict=True
                ),
                'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
                'f1_score': f1_score(y_test, y_pred),
                'cv_f1_mean': np.nanmean(cv_scores),
                'cv_f1_std': np.nanstd(cv_scores),
                'model': model
            }

            # manual confusion matrix plot
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            disp = ConfusionMatrixDisplay(cm, display_labels=['Beneficially','Risky'])
            fig, ax = plt.subplots(figsize=(7,6))
            disp.plot(cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix – {name}')
            plt.tight_layout()
            plt.show()

            # ROC curve plot
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig, ax = plt.subplots(figsize=(7,6))
                ax.plot(fpr, tpr, label=f'AUC = {auc(fpr,tpr):.2f}')
                ax.plot([0,1],[0,1],'k--')
                ax.set_title(f'ROC Curve – {name}')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc='lower right')
                plt.tight_layout()
                plt.show()

        return models, results

    except Exception as e:
        print(f"Error training models: {e}")
        return None, None


def evaluate_overfitting(models, X_train, X_test, y_train, y_test):
    report = {}
    for name, model in models.items():
        try:
            ta = model.score(X_train, y_train)
            te = model.score(X_test, y_test)
            report[name] = {
                'train_accuracy': round(ta,4),
                'test_accuracy': round(te,4),
                'overfit_ratio': round(ta-te,4)
            }
            print(f"{name}: train={ta:.3f}, test={te:.3f}, overfit={ta-te:.3f}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    return report
