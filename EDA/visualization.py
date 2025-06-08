import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from .file_operations import select_save_directory, save_plot

# Visualize insights and model performance
def visualize_data(df, models_results=None):
    try:
        save_dir = select_save_directory()
        df_viz = df.copy()
        df_viz['Behavior_Risk_Label'] = df_viz['Behavior_Risk_Level'] \
            .map({0: 'Beneficially', 1: 'Risky'})

        # Risk level count plot
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

        # Boxplot of key risk-related features
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

        # Correlation heatmap of top numeric features
        numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
        corr = df_viz[numeric_cols].corr()
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

        # Learning curve for logistic regression
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

        # Confusion matrix and ROC for each model
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

        return df_viz, save_dir

    except Exception as e:
        print(f"Error in visualizations: {e}")
        return None