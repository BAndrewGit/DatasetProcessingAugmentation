import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, ConfusionMatrixDisplay, silhouette_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import json
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pd.set_option('future.no_silent_downcasting', True)

# Configurare parametri
CONFIG = {
    'required_columns': [
        'Age',
        'Essential_Needs_Percentage',
        'Expense_Distribution_Entertainment',
        'Debt_Level',
        'Savings_Goal_Emergency_Fund',
        'Behavior_Risk_Level'
    ],
    'risk_weights': [0.3, 0.25, 0.35, 0.1],
    'test_size': 0.25,
    'kmeans_clusters': 3,
    'dynamic_threshold': 'median',  # 'mean', 'quantile' sau valoare fixă
    'min_samples_cluster': 10,
    'max_na_percentage': 0.2,
}

###############################
# Funcții de încărcare și preprocesare
###############################

def load_data():
    """Încarcă fișierul CSV prin dialog."""
    try:
        Tk().withdraw()
        file_path = filedialog.askopenfilename(
            title="Selectează fișierul CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            print("Operațiune anulată")
            return None

        df = pd.read_csv(file_path)

        # Verifică coloanele obligatorii
        missing_cols = [col for col in CONFIG['required_columns'] if col not in df.columns]
        if missing_cols:
            print(f"Fișierului îi lipsesc coloanele: {', '.join(missing_cols)}")
            return None

        return df

    except Exception as e:
        print(f"Eroare la încărcare fișier: {str(e)}")
        return None


def preprocess_data(df):
    """
    Preprocesare avansată:
      - Aplica conversia duratelor.
      - Impută valori lipsă.
      - Converteste variabilele categorice (ex. ordinal și nominal).
      - Pentru variabilele nominale care au sub 10 valori unice, se aplică one-hot encoding.
      - Se returnează DataFrame-ul preprocesat.
    """
    try:
        savings_goal_cols = ['Savings_Goal_Emergency_Fund', 'Savings_Goal_Major_Purchases',
                             'Savings_Goal_Child_Education', 'Savings_Goal_Vacation',
                             'Savings_Goal_Retirement', 'Savings_Goal_Other']

        savings_obstacle_cols = ['Savings_Obstacle_Insufficient_Income', 'Savings_Obstacle_Other_Expenses',
                                 'Savings_Obstacle_Not_Priority', 'Savings_Obstacle_Other']

        expense_dist_cols = ['Expense_Distribution_Food', 'Expense_Distribution_Housing',
                             'Expense_Distribution_Transport', 'Expense_Distribution_Entertainment',
                             'Expense_Distribution_Health', 'Expense_Distribution_Personal_Care',
                             'Expense_Distribution_Child_Education', 'Expense_Distribution_Other']

        credit_cols = ['Credit_Essential_Needs', 'Credit_Major_Purchases',
                       'Credit_Unexpected_Expenses', 'Credit_Personal_Needs', 'Credit_Never_Used']

        passthrough_cols = savings_goal_cols + savings_obstacle_cols + expense_dist_cols + credit_cols

        # Coloanele pentru durate
        duration_cols = ['Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
                         'Product_Lifetime_Appliances', 'Product_Lifetime_Cars']
        def convert_duration(s):
            if isinstance(s, str):
                if 'month' in s:
                    return int(s.split()[0])
                elif 'year' in s:
                    return int(s.split()[0]) * 12
            return np.nan
        for col in duration_cols:
            df[col] = df[col].apply(convert_duration)

        # Seturi de coloane
        numerical_cols = ['Age', 'Income_Category', 'Essential_Needs_Percentage'] + duration_cols
        ordinal_cols = ['Debt_Level', 'Impulse_Buying_Frequency']
        nominal_cols = ['Family_Status', 'Gender', 'Financial_Attitude',
                        'Budget_Planning', 'Save_Money',
                        'Impulse_Buying_Category', 'Impulse_Buying_Reason',
                        'Financial_Investments', 'Bank_Account_Analysis_Frequency']

        # Imputare și conversie pentru variabile numerice
        for col in numerical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # Preprocesare pentru variabile categorice
        for col in ordinal_cols + nominal_cols:
            if col == 'Debt_Level':
                df[col] = df[col].fillna("Absent")
            else:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = df[col].astype(str)

        # Definim categoriile ordinale pentru Debt_Level
        debt_level_categories = ['Absent', 'Low', 'Manageable', 'Difficult to manage']
        df['Debt_Level'] = pd.Categorical(df['Debt_Level'], categories=debt_level_categories, ordered=True)
        df['Debt_Level'] = df['Debt_Level'].cat.codes  # Convertim în valori numerice

        # Pentru variabilele nominale, aplicăm one-hot encoding doar dacă au sub 10 valori unice
        nominal_to_dummy = [col for col in nominal_cols if df[col].nunique() < 10]
        for col in nominal_to_dummy:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
            # Eliminăm coloana din lista nominal_cols
            nominal_cols.remove(col)

        # Procesare pentru coloanele credit (binarizare)
        credit_cols_in_df = [col for col in df.columns if col.startswith('Credit_')]
        for col in credit_cols_in_df:
            df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)

        # Convertire la numeric pentru coloanele rămase
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except:
                df.drop(columns=[col], inplace=True)

        return df

    except Exception as e:
        print(f"Eroare gravă la preprocesare: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def load_and_preprocess_data():
    df = load_data()
    if df is None:
        return None
    df = preprocess_data(df)
    return df


###############################
# Funcții pentru modelare
###############################

def train_models(df):
    """
    Antrenează mai mulți algoritmi de predicție (Logistic Regression, Random Forest, SVM) și folosește validare încrucișată.
    Returnează un dicționar cu modelele antrenate și rezultatele evaluării.
    """
    try:
        X = df.drop(columns=['Behavior_Risk_Level'])
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
            ),
            'SVM': None  # Inițializăm ca None, vom construi dinamic
        }

        # Antrenare SVM cu gestionare avertismente
        print("\nAntrenare SVM...")
        with warnings.catch_warnings(record=True) as w:
            svm_model = SVC(
                kernel='linear',
                probability=True,
                class_weight='balanced',
                random_state=42,
                max_iter=500  # Inițial setăm la 500
            )
            svm_model.fit(X_train, y_train)

            # Verificăm dacă există avertismente de convergență
            if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                print("⚠️ SVM nu a convergent! Creștem max_iter la 2000...")
                svm_model = SVC(
                    kernel='linear',
                    probability=True,
                    class_weight='balanced',
                    random_state=42,
                    max_iter=2000
                )
                svm_model.fit(X_train, y_train)

                # Verificăm din nou
                with warnings.catch_warnings(record=True) as w2:
                    svm_model.fit(X_train, y_train)
                    if any(issubclass(warn.category, ConvergenceWarning) for warn in w2):
                        print("⚠️ SVM tot nu converge, folosim SGDClassifier...")
                        from sklearn.linear_model import SGDClassifier
                        svm_model = SGDClassifier(
                            loss='hinge',
                            max_iter=10000,
                            class_weight='balanced',
                            random_state=42,
                            tol=1e-3
                        )
                        svm_model.fit(X_train, y_train)

        models['SVM'] = svm_model

        results = {}

        for name, model in models.items():
            print(f"\nAntrenare {name}...")
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

            # Plot matrice de confuzie
            disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, normalize='true'),
                                          display_labels=['Beneficially', 'Risky'])
            disp.plot(cmap='Blues')
            plt.title(f'Matrice de confuzie - {name}')
            plt.show()

            # Plot ROC curve (dacă modelul poate genera probabilități)
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
        print(f"Eroare la antrenare modele: {str(e)}")
        return None, None, None, None, None, None


###############################
# Funcții pentru vizualizări suplimentare
###############################

def visualize_data(df):
    """
    Generează vizualizări suplimentare:
      1. Distribuția nivelului de risc.
      2. Distribuția factorilor de risc (ex: Essential_Needs_Percentage, Expense_Distribution_Entertainment, Debt_Level).
      3. Pairplot pentru a explora relațiile dintre factorii de risc.
      4. Heatmap de corelații.
      5. Curba de învățare pentru Logistic Regression.
      6. Clustering KMeans.
    """
    try:
        df_viz = df.copy()
        df_viz['Behavior_Risk_Label'] = df_viz['Behavior_Risk_Level'].map({0: 'Beneficially', 1: 'Risky'})

        # 1. Plot distribuție nivel de risc
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Behavior_Risk_Label', data=df_viz, hue='Behavior_Risk_Label', legend=False, palette='Set2')
        plt.title('Distribuția nivelului de risc')
        plt.xlabel('Nivel de risc')
        plt.ylabel('Număr de cazuri')
        plt.show()

        # 2. Plot distribuție factori de risc
        plt.figure(figsize=(10, 6))
        risk_factors = ['Essential_Needs_Percentage', 'Expense_Distribution_Entertainment', 'Debt_Level']
        df_risk = df_viz[risk_factors + ['Behavior_Risk_Label']]
        df_risk.melt(id_vars='Behavior_Risk_Label', var_name='Factor', value_name='Valoare') \
            .pipe(sns.boxplot, x='Factor', y='Valoare', hue='Behavior_Risk_Label', palette='Set3')
        plt.title('Distribuția factorilor de risc')
        plt.xlabel('Factor')
        plt.ylabel('Valoare')
        plt.legend(title='Nivel de risc')
        plt.show()

        # 3. Pairplot pentru factori de risc
        if len(df_viz['Behavior_Risk_Label'].unique()) > 1:
            sns.pairplot(df_viz, vars=risk_factors, hue='Behavior_Risk_Label', palette='husl')
            plt.suptitle('Pairplot: Relații între factorii de risc', y=1.02)
            plt.show()

        # 4. Heatmap de corelații
        plt.figure(figsize=(12, 10))
        corr = df_viz.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Heatmap Corelații între Caracteristici')
        plt.show()

        # 5. Curba de învățare pentru Logistic Regression
        from sklearn.linear_model import LogisticRegression
        X_lr = df_viz.drop(columns=['Behavior_Risk_Level', 'Behavior_Risk_Label'])
        y_lr = df_viz['Behavior_Risk_Level']
        train_sizes, train_scores, test_scores = learning_curve(
            LogisticRegression(max_iter=1000, solver='newton-cg', random_state=42),
            X_lr, y_lr, cv=5, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 5)
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train F1 score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='CV F1 score')
        plt.title('Curba de învățare pentru Logistic Regression')
        plt.xlabel('Număr de mostre de antrenament')
        plt.ylabel('Scor F1')
        plt.legend(loc='best')
        plt.show()

        # 6. Clustering cu KMeans
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(df_viz[['Essential_Needs_Percentage', 'Expense_Distribution_Entertainment']])
        if X_cluster.shape[0] >= CONFIG['min_samples_cluster']:
            kmeans = KMeans(n_clusters=CONFIG['kmeans_clusters'], random_state=42)
            clusters = kmeans.fit_predict(X_cluster)
            df_viz['Cluster'] = clusters

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=X_cluster[:, 0], y=X_cluster[:, 1], hue=clusters, palette='viridis',
                            style=df_viz['Behavior_Risk_Label'], s=100)
            plt.xlabel('Essential_Needs_Percentage (standardizat)')
            plt.ylabel('Expense_Distribution_Entertainment (standardizat)')
            plt.title('Clustering KMeans: Necesități vs. Cheltuieli divertisment')
            plt.show()

            sil_score = silhouette_score(X_cluster, clusters)
            print(f"Silhouette Score: {sil_score:.2f}")
        else:
            print(f"Prea puține mostre ({X_cluster.shape[0]}) pentru clustering.")

        return df_viz

    except Exception as e:
        print(f"Eroare la vizualizări: {str(e)}")
        return None


def save_metrics(results):
    """Salvează metricile în fișier JSON, inclusiv rapoartele pentru toate modelele."""
    try:
        metrics = {}
        for name, res in results.items():
            metrics[name] = {
                'classification_report': res['classification_report'],
                'roc_auc': res['roc_auc'],
                'f1_score': res['f1_score'],
                'cv_f1_mean': res['cv_f1_mean'],
                'cv_f1_std': res['cv_f1_std']
            }

        with open('dataset_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print("Metricile au fost salvate cu succes în 'dataset_metrics.json'.")

    except Exception as e:
        print(f"Eroare la salvarea metricilor: {str(e)}")


###############################
# Funcția main – fluxul complet
###############################

def main():
    df = load_and_preprocess_data()
    if df is None:
        return

    models, results, X_train, X_test, y_train, y_test = train_models(df)
    if models is None:
        return

    df_viz = visualize_data(df)
    save_metrics(results)


if __name__ == "__main__":
    main()
