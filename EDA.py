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
import json
pd.set_option('future.no_silent_downcasting', True)
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
    'dynamic_threshold': 'median',  # 'mean', 'quantile' sau valoare fixă
    'min_samples_cluster': 10,
    'max_na_percentage': 0.2
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

        # Verificare scoruri unice
        if df['Risk_Score'].nunique() == 1:
            print("\nToate scorurile de risc sunt identice! Ajustați ponderile.")
            return None

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
    try:
        # Folosește toate coloanele, cu excepția variabilei țintă
        X = df.drop(columns=['Behavior_Risk_Level'])  # Exclude doar coloana țintă
        y = df['Behavior_Risk_Level']

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

        # Analiza importanței caracteristicilor
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': logreg.coef_[0]
        })
        print("\nImportanța caracteristicilor:")
        print(feature_importance.sort_values(by='Coefficient', ascending=False))

        return logreg, y_test, y_pred

    except Exception as e:
        print(f"Eroare la antrenare modele: {str(e)}")
        return None, None, None


def preprocess_data(df):
    """Preprocesare avansată a tuturor coloanelor"""
    try:
        # 4. Procesare coloane categorice ordinale
        ordinal_mappings = {
            'Impulse_Buying_Frequency': {
                'Very rarely': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Very often': 5
            },
            'Debt_Level': {
                'Difficult to manage': 3, 'Manageable': 2, 'Low': 1, 'None': 0, np.nan: 0, 'Unknown': 0
            },
            'Bank_Account_Analysis_Frequency': {
                'Rarely or never': 1, 'Monthly': 2, 'Weekly': 3, 'Daily': 4
            }
        }

        for col, mapping in ordinal_mappings.items():
            df[col] = df[col].map(mapping).fillna(0).astype(int)

        # 6. Procesare coloane categorice nominale
        nominal_cols = [
            'Family_Status', 'Gender', 'Financial_Attitude', 'Budget_Planning',
            'Save_Money', 'Impulse_Buying_Category', 'Impulse_Buying_Reason',
            'Financial_Investments', 'Savings_Obstacle'
        ]

        # Filtrăm doar coloanele existente
        nominal_cols = [col for col in nominal_cols if col in df.columns]

        # One-hot encoding pentru categorii cu sub 10 valori unice
        for col in nominal_cols:
            if df[col].nunique() < 10:
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)

        # 7. Procesare coloane Credit_*
        credit_cols = [c for c in df.columns if c.startswith('Credit_')]
        for col in credit_cols:
            df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)

        # 8. Conversia coloanelor rămase la numeric
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except:
                # Dacă nu poate fi convertită, eliminăm coloana
                df.drop(columns=[col], inplace=True)

        # 9. Imputare valori lipsă
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        return df

    except Exception as e:
        print(f"Eroare gravă la preprocesare: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


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

        # Verificare număr minim de mostre
        if X_scaled.shape[0] < CONFIG['min_samples_cluster']:
            print(f"\nPrea puține mostre ({X_scaled.shape[0]}) pentru clustering.")
            return None, None

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

        return X_scaled, clusters

    except Exception as e:
        print(f"Eroare la vizualizări: {str(e)}")
        return None, None


def save_metrics(df, model, y_test, y_pred, X_scaled, clusters):
    """Salvează metricile în fișier JSON"""
    try:
        metrics = {
            'class_distribution': df['Behavior_Risk_Level'].value_counts(normalize=True).to_dict(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'silhouette_score': round(silhouette_score(X_scaled, clusters),
                                      2) if X_scaled is not None and clusters is not None else None,
            'logreg_coefficients': {col: round(coef, 3) for col, coef in
                                    zip(model.feature_names_in_, model.coef_[0])} if model else None
        }

        with open('dataset_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    except Exception as e:
        print(f"Eroare la salvarea metricilor: {str(e)}")


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

    model, y_test, y_pred = train_models(df)
    if model is None:
        return

    X_scaled, clusters = visualize_data(df)
    save_metrics(df, model, y_test, y_pred, X_scaled, clusters)


if __name__ == "__main__":
    main()