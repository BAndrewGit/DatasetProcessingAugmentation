import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def main():
    # 1. Încărcare date
    df = pd.read_csv('DatasetOriginal.csv')

    # 2. Creare și ajustare metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    for column in df.columns:
        if df[column].dtype == 'object':
            metadata.update_column(column_name=column, sdtype='categorical')
        else:
            metadata.update_column(column_name=column, sdtype='numerical')

    # 3. Split dataset
    X = df.drop(columns=['Behavior_Risk_Level'])
    y = df['Behavior_Risk_Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # 4. Aplică ADASYN
    adasyn = ADASYN(random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)

    # 5. Creează DataFrame combinat pentru CTGAN
    data_combined = pd.concat([X_adasyn, y_adasyn], axis=1)

    # 6. Identifică coloane categorice
    categorical_cols = data_combined.select_dtypes(include=['object', 'category']).columns.tolist()

    # 7. Inițializează și antrenează CTGAN
    ctgan = CTGANSynthesizer(
        metadata=metadata,
        epochs=200,
        batch_size=500,
        verbose=True,
        cuda=True  # Dacă aveți GPU
    )
    ctgan.fit(data_combined, categorical_cols)

    # 8. Generează și filtrează mostre sintetice
    synthetic_samples = ctgan.sample(num_rows=len(X_adasyn) // 2)
    synthetic_samples = synthetic_samples.dropna()  # Elimină rândurile goale

    # 9. Combina datele
    X_final = pd.concat([X_adasyn, synthetic_samples.drop(columns=['Behavior_Risk_Level'])])
    y_final = pd.concat([y_adasyn, synthetic_samples['Behavior_Risk_Level']])

    # 10. Salvarea rezultatelor
    X_final.to_csv('ADASYN_WCGAN_augmented.csv', index=False)
    y_final.to_csv('ADASYN_WCGAN_labels.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('ADASYN_WCGAN_test_set.csv', index=False)

    with pd.ExcelWriter('ADASYN_WCGAN_augmented.xlsx') as writer:
        X_final.to_excel(writer, sheet_name='Date_Augmentate', index=False)
        y_final.to_excel(writer, sheet_name='Etichete', index=False)
        pd.concat([X_test, y_test], axis=1).to_excel(writer, sheet_name='Test_Set', index=False)

if __name__ == "__main__":
    main()