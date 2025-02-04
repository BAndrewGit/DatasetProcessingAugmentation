import pandas as pd
from sklearn.model_selection import train_test_split
from sdv.single_table import CTGANSynthesizer


def main():
    df = pd.read_csv('processed_dataset.csv')
    X = df.drop(columns=['Behavior_Risk_Level'])
    y = df['Behavior_Risk_Level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    train_data = pd.concat([X_train, y_train], axis=1)

    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist() + ['Behavior_Risk_Level']

    ctgan = CTGANSynthesizer(epochs=150)
    ctgan.fit(train_data, categorical_cols)

    synthetic_samples = ctgan.sample(1000)
    synthetic_samples = synthetic_samples[synthetic_samples['Behavior_Risk_Level'] == 'Risky']

    X_final = pd.concat([X_train, synthetic_samples.drop(columns=['Behavior_Risk_Level'])])
    y_final = pd.concat([y_train, synthetic_samples['Behavior_Risk_Level']])

    # Salvarea în CSV și Excel
    X_final.to_csv('CGAN_augmented.csv', index=False)
    y_final.to_csv('CGAN_labels.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('CGAN_test_set.csv', index=False)

    with pd.ExcelWriter('CGAN_augmented.xlsx') as writer:
        X_final.to_excel(writer, sheet_name='Date_Augmentate', index=False)
        y_final.to_excel(writer, sheet_name='Etichete', index=False)
        pd.concat([X_test, y_test], axis=1).to_excel(writer, sheet_name='Test_Set', index=False)


if __name__ == "__main__":
    main()