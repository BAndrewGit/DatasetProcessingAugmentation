import pandas as pd
from FirstProcessing.preprocessing import range_smoothing

# Columns expected for modeling
CONFIG = {
    'required_columns': [
        'Age', 'Essential_Needs_Percentage', 'Expense_Distribution_Entertainment',
        'Debt_Level', 'Save_Money_Yes', 'Behavior_Risk_Level'
    ]
}

# Fill missing values in required columns
def preprocess_encoded_data(df):
    for col in CONFIG['required_columns']:
        if col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
    return df