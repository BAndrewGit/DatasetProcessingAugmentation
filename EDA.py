import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import sys
import json

# Configurare parametri
CONFIG = {
    'required_columns': [
        'Age',
        'Essential_Needs_Percentage',
        'Expense_Distribution_Entertainment',
        'Debt_Level',
        'Savings_Goal_Emergency_Fund'
    ],
    'risk_weights': [0.3, 0.25, 0.35, 0.1],
    'test_size': 0.25,
    'kmeans_clusters': 3,
    'dynamic_threshold': 'median'  # 'mean', 'quantile' sau valoare fixă
}


def load_data():
    """Încarcă fișierul CSV prin dialog"""
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
    """Preprocesare avansată a datelor"""
    try:
        # 1. Transformare Age
        def parse_age(age_str):
            if pd.isna(age_str):
                return np.nan
            if '-' in age_str:
                parts = list(map(int, age_str.split('-')))
                return np.mean(parts)
            elif age_str.startswith('>'):
                return int(age_str[1:]) + 5
            elif age_str.startswith('<'):
                return int(age_str[1:]) - 5
            else:
                return float(age_str) if age_str.replace('.', '', 1).isdigit() else np.nan

        df['Age'] = df['Age'].apply(parse_age)

        # 2. Essential_Needs_Percentage
        df['Essential_Needs_Percentage'] = pd.to_numeric(
            df['Essential_Needs_Percentage'].astype(str).str.replace('[^0-9.]', '', regex=True),
            errors='coerce'
        )

        # 3. Debt_Level cu gestionare NaN
        debt_mapping = {
            'Difficult to manage': 3,
            'Manageable': 2,
            'Low': 1,
            np.nan: 0,
            'Unknown': 0
        }
        df['Debt_Level'] = df['Debt_Level'].replace(debt_mapping).fillna(0).astype(int)

        # 4. Alte coloane numerice
        numeric_cols = ['Expense_Distribution_Entertainment', 'Savings_Goal_Emergency_Fund']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Imputare valori lipsă
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Essential_Needs_Percentage'] = df['Essential_Needs_Percentage'].fillna(
            df['Essential_Needs_Percentage'].median()
        )

        return df

    except Exception as e:
        print(f"Eroare la preprocesare: {str(e)}")
        return None


def calculate_risk_score(df):
    """Calculează scorul de risc și etichetele"""
    try:
        # Calcul condiții cu verificare dimensiuni
        conditions = [
            (df['Essential_Needs_Percentage'] < 45) * CONFIG['risk_weights'][0],
            (df['Expense_Distribution_Entertainment'] > 25) * CONFIG['risk_weights'][1],
            (df['Debt_Level'] >= 2) * CONFIG['risk_weights'][2],
            (df['Savings_Goal_Emergency_Fund'] == 0) * CONFIG['risk_weights'][3]
        ]

        df['Risk_Score'] = np.sum(conditions, axis=0)

        # Determinare prag dinamic
        if isinstance(CONFIG['dynamic_threshold'], (int, float)):
            threshold = CONFIG['dynamic_threshold']
        elif CONFIG['dynamic_threshold'] == 'median':
            threshold = df['Risk_Score'].median()
        elif CONFIG['dynamic_threshold'] == 'mean':
            threshold = df['Risk_Score'].mean()
        else:
            threshold = df['Risk_Score'].quantile(0.75)

        print(f"\nPrag automat determinat: {threshold:.2f}")

        df['Behavior_Risk_Level'] = np.where(
            df['Risk_Score'] > threshold,
            'Riscant',
            'Benefic'
        )

        return df

    except Exception as e:
        print(f"Eroare la calcul risc: {str(e)}")
        return None


def train_models(df):
    """Antrenează modelele de ML"""
    try:
        X = df[['Essential_Needs_Percentage',
                'Expense_Distribution_Entertainment',
                'Debt_Level']]
        y = df['Behavior_Risk_Level']

        # Verificare clase
        unique_classes = y.unique()
        if len(unique_classes) == 1:
            print("\nATENȚIE: Toate instanțele sunt de același tip!")
            print("Nu se poate antrena modelul. Verificați calculul riscului.")
            return None, None

        # Împărțire date
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=CONFIG['test_size'],
            stratify=y,
            random_state=42
        )

        # Logistic Regression
        logreg = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        logreg.fit(X_train, y_train)

        # Evaluare
        y_pred = logreg.predict(X_test)
        print("\nRaport clasificare:")
        print(classification_report(y_test, y_pred, target_names=['Benefic', 'Riscant']))

        # Afișare matrice confuzie
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=['Benefic', 'Riscant'],
            cmap='Blues'
        )
        plt.title('Matrice de confuzie')
        plt.show()

        return logreg, y_test

    except Exception as e:
        print(f"Eroare la antrenare modele: {str(e)}")
        return None, None


def visualize_data(df):
    """Generează vizualizări"""
    try:
        # Distribuție risc
        risk_dist = df['Behavior_Risk_Level'].value_counts(normalize=True)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=risk_dist.index, y=risk_dist.values)
        plt.title('Distribuția nivelului de risc')
        plt.ylabel('Procentaj')
        plt.show()

        # Pairplot doar dacă avem ambele clase
        if len(df['Behavior_Risk_Level'].unique()) > 1:
            sns.pairplot(
                df,
                vars=['Essential_Needs_Percentage',
                      'Expense_Distribution_Entertainment',
                      'Debt_Level'],
                hue='Behavior_Risk_Level',
                palette='husl'
            )
            plt.suptitle('Analiza multivariabilă a factorilor de risc', y=1.02)
            plt.show()

        # Clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[['Essential_Needs_Percentage',
                                            'Expense_Distribution_Entertainment']])

        kmeans = KMeans(n_clusters=CONFIG['kmeans_clusters'], random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=X_scaled[:, 0],
            y=X_scaled[:, 1],
            hue=clusters,
            style=df['Behavior_Risk_Level'],
            palette='viridis'
        )
        plt.xlabel('Necesități esențiale (standardizate)')
        plt.ylabel('Cheltuieli divertisment (standardizate)')
        plt.title('Clustering cu K-means și suprapunere risc')
        plt.show()

    except Exception as e:
        print(f"Eroare la vizualizări: {str(e)}")


def save_metrics(df, y_test, y_pred, X_scaled, clusters):
    metrics = {
        'class_distribution': df['Behavior_Risk_Level'].value_counts(normalize=True).to_dict(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'silhouette_score': round(silhouette_score(X_scaled, clusters), 2),
        'logreg_coefficients': {col: round(coef, 3) for col, coef in zip(X.columns, logreg.coef_[0])}
    }

    with open('dataset_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    """Flux principal"""
    df = load_data()
    if df is None:
        return

    df = preprocess_data(df)
    if df is None:
        return

    df = calculate_risk_score(df)
    if df is None:
        return

    print("\nDistribuție inițială risc:")
    print(df['Behavior_Risk_Level'].value_counts(dropna=False))

    if len(df['Behavior_Risk_Level'].unique()) == 1:
        print("\nNu există suficiente variante de risc pentru analiză!")
        print("Posibile soluții:")
        print("- Ajustați ponderile din CONFIG['risk_weights']")
        print("- Modificați CONFIG['dynamic_threshold']")
        return

    model, y_test = train_models(df)
    if model is None:
        return

    visualize_data(df)
    save_metrics(df, y_test, y_pred, X_scaled, clusters)

if __name__ == "__main__":
    main()