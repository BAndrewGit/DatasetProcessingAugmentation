#!/usr/bin/env python3
import os
import json
import re
import pandas as pd
from pathlib import Path
from tkinter import Tk, filedialog

from FirstProcessing.preprocessing import (
    normalize_and_translate_data,
    postprocess_data,
    range_smoothing
)
from FirstProcessing.risk_calculation import apply_existing_scaler
from FirstProcessing.data_generation import random_product_lifetime
from FirstProcessing.file_operations import save_files

# Pipeline settings
multi_value_cols = [
    "Savings_Goal",
    "Savings_Obstacle",
    "Expense_Distribution",
    "Credit_Usage"
]
lifetime_cols = [
    "Product_Lifetime_Clothing",
    "Product_Lifetime_Tech",
    "Product_Lifetime_Appliances",
    "Product_Lifetime_Cars"
]
numeric_cols_to_scale = [
    'Age', 'Income_Category', 'Essential_Needs_Percentage',
    'Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
    'Product_Lifetime_Appliances', 'Product_Lifetime_Cars'
]
# Paths to saved scaler and dummy schema
scaler_path = Path("scaler/robust_scaler.pkl")
dummy_cols_path = Path("scaler/dummy_columns.json")

# Utility: convert 'X years'/'Y months' to float months

def convert_duration_to_months(value):
    if isinstance(value, str):
        m = re.match(r"(\d+(?:\.\d+)?)\s*years?", value)
        if m:
            return float(m.group(1)) * 12.0
        m2 = re.match(r"(\d+(?:\.\d+)?)\s*months?", value)
        if m2:
            return float(m2.group(1))
        return 0.0
    try:
        return float(value)
    except:
        return 0.0


def main():
    Tk().withdraw()
    # 1) Select decoded input (Excel)
    input_path = filedialog.askopenfilename(
        title="Select decoded Excel file",
        filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"),("All files","*.*")]
    )
    if not input_path:
        print("❌ No file selected. Exiting.")
        return

    # 2) Load data
    try:
        if input_path.lower().endswith('.xlsx'):
            df = pd.read_excel(input_path, engine='openpyxl')
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return

    # 3) Normalize & translate textual responses
    df = normalize_and_translate_data(df)

    # 4) Range smoothing for lifetimes and categories
    #    Use default (decoded) smoothing to convert strings to numeric
    df = range_smoothing(
        df,
        age_column="Age",
        income_column="Income_Category",
        lifetime_columns=lifetime_cols,
        essential_needs_column="Essential_Needs_Percentage",
        lifetime_func=random_product_lifetime
    )

    # 5) Ensure any leftover durations are numeric months
    for col in lifetime_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_duration_to_months)

    # 6) Post-process: ordinal mappings and one-hot for nominal
    df = postprocess_data(df)
    if df is None:
        print("❌ postprocess_data returned None. Aborting.")
        return

    # 7) Reindex dummy columns to match training schema
    if not dummy_cols_path.exists():
        print(f"❌ Dummy schema not found: {dummy_cols_path}")
        return
    with open(dummy_cols_path, 'r') as f:
        dummy_cols = json.load(f)
    # Generate full dummies from current df
    df = pd.get_dummies(df, drop_first=False)
    # Combine non-dummy and dummy in fixed order
    non_dummy = [c for c in df.columns if c not in dummy_cols]
    df = df.reindex(columns=non_dummy + dummy_cols, fill_value=0)

    # 8) Apply existing RobustScaler (no re-fit)
    if not scaler_path.exists():
        print(f"❌ Scaler not found at {scaler_path}. Run training pipeline first.")
        return
    df = apply_existing_scaler(df, numeric_cols_to_scale, scaler_path)

    # 9) Save results (Excel + CSV)
    save_files(df)

if __name__ == '__main__':
    main()
