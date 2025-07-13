import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from tkinter import Tk, filedialog
import os

# Dicționar pentru conversia unităților de timp în luni
TIME_UNITS = {
    'months': 1,
    'month': 1,
    'years': 12,
    'year': 12,
    'ani': 12,
    'an': 12,
    'luni': 1
}


def convert_lifetime_to_months(value):
    if pd.isna(value) or value == 'Not purchased yet':
        return np.nan

    if isinstance(value, (int, float)):
        return value * 12  # Presupunem că valorile numerice sunt în ani

    try:
        return float(value) * 12
    except ValueError:
        pass

    match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z]+)', str(value))
    if match:
        num = float(match.group(1))
        unit = match.group(2).lower()
        multiplier = TIME_UNITS.get(unit, 1)
        return num * multiplier

    return np.nan


ORDINAL_MAPPINGS = {
    'Impulse_Buying_Frequency': {
        'Very rarely': 1,
        'Rarely': 2,
        'Sometimes': 3,
        'Often': 4,
        'Very often': 5
    },
    'Debt_Level': {
        'Absent': 1,
        'Low': 2,
        'Manageable': 3,
        'Difficult to manage': 4
    },
    'Bank_Account_Analysis_Frequency': {
        'Rarely or never': 1,
        'Monthly': 2,
        'Weekly': 3,
        'Daily': 4
    }
}


def convert_decoded_excel_to_encoded():
    root = Tk()
    root.withdraw()

    excel_path = filedialog.askopenfilename(
        title="Selectați fișierul Excel decodat",
        filetypes=[("Excel files", "*.xlsx")]
    )
    if not excel_path:
        print("Nu a fost selectat niciun fișier.")
        return

    try:
        df = pd.read_excel(excel_path, sheet_name='Decoded_Data')

        # 1. Conversia duratei de viață a produselor
        lifetime_cols = [
            'Product_Lifetime_Clothing',
            'Product_Lifetime_Tech',
            'Product_Lifetime_Appliances',
            'Product_Lifetime_Cars'
        ]

        for col in lifetime_cols:
            df[col] = df[col].apply(convert_lifetime_to_months)
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            df[col] = df[col].round().astype(int)

        # 2. Conversia riscului comportamental
        df['Behavior_Risk_Level'] = df['Behavior_Risk_Level'].map({
            'Beneficial': 0,
            'Risky': 1
        })

        # 3. Codificare ordinală
        for col, mapping in ORDINAL_MAPPINGS.items():
            df[col] = df[col].map(mapping)
            df[col].fillna(0, inplace=True)
            df[col] = df[col].astype(int)

        # 4. One-hot encoding doar pentru coloanele nominale existente
        nominal_cols = [
            'Family_Status',
            'Gender',
            'Financial_Attitude',
            'Budget_Planning',
            'Save_Money',
            'Impulse_Buying_Category',
            'Impulse_Buying_Reason',
            'Financial_Investments'
        ]

        for col in nominal_cols:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            df = pd.concat([df, dummies], axis=1)

        df.drop(columns=nominal_cols, inplace=True)

        # 5. Scalare numerică
        numeric_cols = ['Age', 'Income_Category', 'Essential_Needs_Percentage'] + lifetime_cols
        scaler_path = Path("scaler/robust_scaler.pkl")

        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        else:
            print(f"Scaler not found at {scaler_path}. Using unscaled data.")

        # 6. Salvarea ca CSV
        base_name = os.path.splitext(excel_path)[0]
        csv_path = f"{base_name}_encoded.csv"
        df.to_csv(csv_path, index=False)

        print(f"Fișierul encoded.csv a fost salvat la: {csv_path}")
        return csv_path

    except Exception as e:
        print(f"Eroare la procesare: {str(e)}")
        return None


if __name__ == "__main__":
    convert_decoded_excel_to_encoded()