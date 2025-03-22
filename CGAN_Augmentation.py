import pandas as pd
from sdv.metadata import SingleTableMetadata
from CTGAN.ctgan.synthesizers.ctgan import CTGAN


# Funcție pentru definirea metadata-ului
def define_metadata(df):
    metadata = SingleTableMetadata()

    # Definirea sdtype pentru fiecare coloană
    column_types = {
        'Age': 'numerical',
        'Family_Status': 'categorical',
        'Gender': 'categorical',
        'Income_Category': 'categorical',
        'Essential_Needs_Percentage': 'numerical',
        'Financial_Attitude': 'categorical',
        'Budget_Planning': 'categorical',
        'Save_Money': 'categorical',
        'Savings_Obstacle': 'categorical',
        'Product_Lifetime_Clothing': 'categorical',
        'Product_Lifetime_Tech': 'categorical',
        'Product_Lifetime_Appliances': 'categorical',
        'Product_Lifetime_Cars': 'categorical',
        'Impulse_Buying_Frequency': 'categorical',
        'Impulse_Buying_Category': 'categorical',
        'Impulse_Buying_Reason': 'categorical',
        'Debt_Level': 'numerical',
        'Financial_Investments': 'categorical',
        'Bank_Account_Analysis_Frequency': 'categorical',
        'Savings_Goal_Emergency_Fund': 'categorical',
        'Savings_Goal_Major_Purchases': 'categorical',
        'Savings_Goal_Child_Education': 'categorical',
        'Savings_Goal_Vacation': 'categorical',
        'Savings_Goal_Retirement': 'categorical',
        'Savings_Goal_Other': 'categorical',
        'Expense_Distribution_Food': 'numerical',
        'Expense_Distribution_Housing': 'numerical',
        'Expense_Distribution_Transport': 'numerical',
        'Expense_Distribution_Entertainment': 'numerical',
        'Expense_Distribution_Health': 'numerical',
        'Expense_Distribution_Personal_Care': 'numerical',
        'Expense_Distribution_Child_Education': 'numerical',
        'Expense_Distribution_Other': 'numerical',
        'Credit_Essential_Needs': 'categorical',
        'Credit_Major_Purchases': 'categorical',
        'Credit_Unexpected_Expenses': 'categorical',
        'Credit_Personal_Needs': 'categorical',
        'Credit_Never_Used': 'categorical',
        'Behavior_Risk_Level': 'categorical'
    }

    # Adăugăm coloanele cu sdtype corespunzător
    for col, col_type in column_types.items():
        metadata.add_column(column_name=col, sdtype=col_type)

    return metadata


# Funcție pentru antrenarea modelului CTGAN și generarea de date augmentate
def generate_synthetic_data(df):
    metadata = define_metadata(df)

    # Crearea unui model CTGAN
    model = CTGAN(metadata)

    # Antrenarea modelului
    model.fit(df)

    # Generarea datelor augmentate
    augmented_data = model.sample()

    return augmented_data


# Funcție principală
def main():
    # Încarcă datele originale dintr-un fișier CSV
    df = pd.read_csv('DatasetOriginal.csv')

    # Generează date augmentate folosind GAN
    augmented_data = generate_synthetic_data(df)

    # Salvează datele augmentate într-un fișier CSV
    augmented_data.to_csv('C-GAN_augmented.csv', index=False)
    print("Datele augmentate au fost salvate în 'C-GAN_augmented.csv'.")


# Apelarea funcției principale
if __name__ == "__main__":
    main()
