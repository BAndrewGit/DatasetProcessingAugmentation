import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from .file_operations import select_save_directory, save_plot

def visualize_data(df, models_results=None):
    try:
        save_dir = select_save_directory()
        df_viz = df.copy()
        df_viz['Behavior_Risk_Label'] = df_viz['Behavior_Risk_Level'].map({0: 'Beneficially', 1: 'Risky'})

        # Risk level count plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            x='Behavior_Risk_Label',
            data=df_viz,
            hue='Behavior_Risk_Label',
            legend=False,
            palette='Set2',
            ax=ax
        )
        ax.set_title('Risk Level Distribution')
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Number of Cases')
        plt.tight_layout()
        save_plot(fig, save_dir, 'risk_level_distribution.png')

        # Boxplot of continuous risk factors
        risk_factors = [
            'Age', 'Income_Category', 'Essential_Needs_Percentage',
            'Debt_Level', 'Product_Lifetime_Clothing',
            'Product_Lifetime_Tech', 'Product_Lifetime_Appliances',
            'Product_Lifetime_Cars'
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

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            x='Factor',
            y='Value',
            hue='Behavior_Risk_Label',
            data=df_melted,
            palette='Set3',
            dodge=True,
            showfliers=False,
            whis=1.5,
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

        # Correlation heatmap
        continuous_cols = risk_factors
        bool_cols = [
            'Credit_Usage_Never_Used',
            'Budget_Planning_Plan only essentials',
            'Impulse_Buying_Category_Entertainment',
            'Family_Status_In a relationship/married with children',
            'Savings_Goal_Emergency_Fund',
            'Financial_Investments_Yes, occasionally',
            'Savings_Goal_Retirement'
        ]
        corr_df = df_viz[continuous_cols].copy()
        corr_df['Behavior_Risk_Level'] = df_viz['Behavior_Risk_Level']
        for col in bool_cols:
            if col in df_viz:
                corr_df[col] = df_viz[col].astype(int)

        corr_matrix = corr_df.corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            cmap='coolwarm',
            annot=True,
            fmt=".2f",
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title('Correlation Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        save_plot(fig, save_dir, 'correlation_heatmap.png')

        # Learning curve for logistic regression
        X_lr = df_viz.drop(columns=['Behavior_Risk_Level', 'Behavior_Risk_Label'])
        y_lr = df_viz['Behavior_Risk_Level']
        train_sizes, train_scores, test_scores = learning_curve(
            LogisticRegression(max_iter=1000, solver='newton-cg', random_state=42),
            X_lr, y_lr,
            cv=5,
            scoring='f1',
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

        # Confusion matrix and ROC for each model (manual)
        if models_results:
            for model_name, (model, X_test, y_test) in models_results.items():
                y_pred = model.predict(X_test)
                # confusion matrix
                cm = confusion_matrix(y_test, y_pred, normalize='true')
                disp = ConfusionMatrixDisplay(cm, display_labels=['Beneficially', 'Risky'])
                fig, ax = plt.subplots(figsize=(7, 6))
                disp.plot(cmap='Blues', ax=ax)
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
        return None, None
