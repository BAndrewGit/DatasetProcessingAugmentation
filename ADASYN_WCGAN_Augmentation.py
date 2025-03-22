import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


def preprocess_data(df):
    categorical_cols = [
        'Family_Status', 'Gender', 'Financial_Attitude', 'Budget_Planning', 'Save_Money',
        'Impulse_Buying_Category', 'Impulse_Buying_Reason', 'Debt_Level', 'Financial_Investments',
        'Bank_Account_Analysis_Frequency', 'Savings_Goal_Emergency_Fund',
        'Savings_Goal_Major_Purchases', 'Savings_Goal_Child_Education', 'Savings_Goal_Vacation',
        'Savings_Goal_Retirement', 'Savings_Goal_Other', 'Savings_Obstacle_Insufficient_Income',
        'Savings_Obstacle_Other_Expenses', 'Savings_Obstacle_Not_Priority', 'Savings_Obstacle_Other',
        'Credit_Essential_Needs', 'Credit_Major_Purchases', 'Credit_Unexpected_Expenses',
        'Credit_Personal_Needs', 'Credit_Never_Used',
        'Impulse_Buying_Frequency',
        'Product_Lifetime_Clothing', 'Product_Lifetime_Tech', 'Product_Lifetime_Appliances', 'Product_Lifetime_Cars'
    ]
    numeric_cols = [
        'Age', 'Income_Category', 'Essential_Needs_Percentage',
        'Expense_Distribution_Food', 'Expense_Distribution_Housing',
        'Expense_Distribution_Transport', 'Expense_Distribution_Entertainment',
        'Expense_Distribution_Health', 'Expense_Distribution_Personal_Care',
        'Expense_Distribution_Child_Education', 'Expense_Distribution_Other'
    ]
    # Pentru numerice: eliminăm sufixele și convertim la numeric
    df[numeric_cols] = df[numeric_cols].astype(str).replace({'months': '', 'years': ''}, regex=True)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    numeric_cols = [col for col in numeric_cols if not df[col].isna().all()]
    # Pentru categorice: completăm valorile lipsă
    df[categorical_cols] = df[categorical_cols].fillna('missing').astype(str)
    return df, categorical_cols, numeric_cols


def safe_int(row, col):
    try:
        return int(float(row.get(col, 0)))
    except Exception:
        return 0


def enforce_constraints(row):
    savings_goal_cols = [
        'Savings_Goal_Emergency_Fund', 'Savings_Goal_Major_Purchases', 'Savings_Goal_Child_Education',
        'Savings_Goal_Vacation', 'Savings_Goal_Retirement', 'Savings_Goal_Other'
    ]
    savings_obstacle_cols = [
        'Savings_Obstacle_Insufficient_Income', 'Savings_Obstacle_Other_Expenses',
        'Savings_Obstacle_Not_Priority', 'Savings_Obstacle_Other'
    ]
    if row.get('Save_Money', 'missing') == "Yes":
        for col in savings_obstacle_cols:
            row[col] = '0'
        if sum([safe_int(row, col) for col in savings_goal_cols]) < 1:
            chosen = np.random.choice(savings_goal_cols)
            row[chosen] = '1'
    elif row.get('Save_Money', 'missing') == "No":
        for col in savings_goal_cols:
            row[col] = '0'
        if sum([safe_int(row, col) for col in savings_obstacle_cols]) != 1:
            for col in savings_obstacle_cols:
                row[col] = '0'
            chosen = np.random.choice(savings_obstacle_cols)
            row[chosen] = '1'

    expense_distribution_cols = [
        'Expense_Distribution_Food', 'Expense_Distribution_Housing', 'Expense_Distribution_Transport',
        'Expense_Distribution_Entertainment', 'Expense_Distribution_Health',
        'Expense_Distribution_Personal_Care', 'Expense_Distribution_Child_Education',
        'Expense_Distribution_Other'
    ]
    if sum([safe_int(row, col) for col in expense_distribution_cols]) != 1:
        for col in expense_distribution_cols:
            row[col] = '0'
        chosen = np.random.choice(expense_distribution_cols)
        row[chosen] = '1'

    credit_cols = [
        'Credit_Essential_Needs', 'Credit_Major_Purchases', 'Credit_Unexpected_Expenses',
        'Credit_Personal_Needs', 'Credit_Never_Used'
    ]
    if sum([safe_int(row, col) for col in credit_cols]) != 1:
        for col in credit_cols:
            row[col] = '0'
        chosen = np.random.choice(credit_cols)
        row[chosen] = '1'

    try:
        age = int(float(row.get('Age', 18)))
    except:
        age = 18
    if age < 18:
        row['Age'] = '18'
    elif age > 100:
        row['Age'] = '100'

    try:
        enp = float(row.get('Essential_Needs_Percentage', 0))
    except:
        enp = 0.0
    if enp < 0:
        row['Essential_Needs_Percentage'] = '0'
    elif enp > 100:
        row['Essential_Needs_Percentage'] = '100'

    try:
        income = float(row.get('Income_Category', 4100))
    except:
        income = 4100
    row['Income_Category'] = str(int(round(income)))

    for col in ['Product_Lifetime_Clothing', 'Product_Lifetime_Tech', 'Product_Lifetime_Appliances',
                'Product_Lifetime_Cars']:
        val = row.get(col, '')
        if isinstance(val, str) and 'months' in val and col == 'Product_Lifetime_Cars':
            try:
                months = int(val.split()[0])
                years = max(1, round(months / 12))
                row[col] = f"{years} years"
            except Exception:
                row[col] = "5 years"

    allowed_family_status = ["Single, no children", "In a relationship/married without children",
                             "In a relationship/married with children"]
    if row.get('Family_Status', '') not in allowed_family_status:
        row['Family_Status'] = allowed_family_status[0]

    allowed_financial_attitude = ["I try to find a balance", "I am disciplined in saving"]
    if row.get('Financial_Attitude', '') not in allowed_financial_attitude:
        row['Financial_Attitude'] = allowed_financial_attitude[0]

    allowed_impulse_reason = ["Self-reward", "Discounts or promotions"]
    if row.get('Impulse_Buying_Reason', '') not in allowed_impulse_reason:
        row['Impulse_Buying_Reason'] = allowed_impulse_reason[0]

    if row.get('Debt_Level', '') == "Absent":
        allowed_financial_investments = ["Yes, occasionally", "No, but interested"]
        if row.get('Financial_Investments', '') not in allowed_financial_investments:
            row['Financial_Investments'] = np.random.choice(allowed_financial_investments)
    elif row.get('Debt_Level', '') in ["Low", "Manageable"]:
        if row.get('Financial_Investments', '') == "Yes, regularly":
            row['Financial_Investments'] = "No, but interested"

    if row.get('Impulse_Buying_Frequency', 'missing') == "Very rarely" and row.get('Impulse_Buying_Reason',
                                                                                   'missing') == "Self-reward":
        row['Impulse_Buying_Reason'] = "Discounts or promotions"

    return row


def apply_constraints(df):
    return df.apply(enforce_constraints, axis=1)


def main():
    df = pd.read_csv('DatasetOriginal.csv')
    df, categorical_cols, numeric_cols = preprocess_data(df)
    X = df[categorical_cols + numeric_cols]
    y = df['Behavior_Risk_Level']

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_pipeline, categorical_cols),
            ('num', num_pipeline, numeric_cols)
        ]
    )
    X_encoded = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)
    adasyn = ADASYN(random_state=42)
    X_adasyn_np, y_adasyn_np = adasyn.fit_resample(X_train, y_train)

    cat_transformer = preprocessor.named_transformers_['cat']
    onehot = cat_transformer.named_steps['onehot']
    n_cat_features = onehot.transform(X[categorical_cols]).shape[1]
    X_adasyn_cat = onehot.inverse_transform(X_adasyn_np[:, :n_cat_features])
    X_adasyn_cat = pd.DataFrame(X_adasyn_cat, columns=categorical_cols)

    num_transformer = preprocessor.named_transformers_['num']
    X_adasyn_num = num_transformer.named_steps['scaler'].inverse_transform(X_adasyn_np[:, n_cat_features:])
    X_adasyn_num = pd.DataFrame(X_adasyn_num, columns=numeric_cols)

    X_adasyn = pd.concat([X_adasyn_cat, X_adasyn_num], axis=1)
    X_adasyn = apply_constraints(X_adasyn)

    data_combined = pd.concat([X_adasyn, pd.Series(y_adasyn_np, name='Behavior_Risk_Level')], axis=1)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data_combined)
    for col in categorical_cols + ['Behavior_Risk_Level']:
        metadata.update_column(column_name=col, sdtype='categorical')

    ctgan = CTGANSynthesizer(metadata=metadata, epochs=200, batch_size=500, verbose=True, cuda=True)
    ctgan.fit(data_combined)
    synthetic_samples = ctgan.sample(num_rows=len(X_adasyn) // 2).dropna()
    synthetic_samples = apply_constraints(synthetic_samples)

    final_dataset = pd.concat([data_combined, synthetic_samples], ignore_index=True)

    # Post-procesare: convertim coloanele numerice la intregi
    for col in ['Age', 'Income_Category', 'Essential_Needs_Percentage',
                'Expense_Distribution_Food', 'Expense_Distribution_Housing',
                'Expense_Distribution_Transport', 'Expense_Distribution_Entertainment',
                'Expense_Distribution_Health', 'Expense_Distribution_Personal_Care',
                'Expense_Distribution_Child_Education', 'Expense_Distribution_Other']:
        final_dataset[col] = final_dataset[col].astype(float).round(0).astype(int)

    final_dataset.to_csv('ADASYN_WCGAN_augmented.csv', index=False)


if __name__ == "__main__":
    main()
