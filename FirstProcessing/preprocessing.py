import pandas as pd
import numpy as np
import re

from FirstProcessing.data_generation import replace_age_column, replace_income_category, \
    replace_product_lifetime_columns, random_product_lifetime, replace_essential_needs


# Translate Romanian column names and values to English
def normalize_and_translate_data(df):
    df.columns = df.columns.str.strip()

    # Remove timestamp column if exists
    if df.columns[0].lower() in ['marcaj de timp', 'timestamp']:
        df.drop(columns=df.columns[0], inplace=True)

    # Rename columns to standard English identifiers
    column_mapping = {
        "Câți ani aveți?": "Age",
        "Care este statutul dumneavoastră familial?": "Family_Status",
        "Care este genul dumneavoastră?": "Gender",
        "În ce categorie se încadrează venitul dumneavoastră lunar?": "Income_Category",
        "Ce procent aproximativ din venit considerați că vă este suficient pentru nevoi esențiale (mâncare, chirie, transport)?": "Essential_Needs_Percentage",
        "Cum ați descrie atitudinea dumneavoastră față de gestionarea banilor?": "Financial_Attitude",
        "Planificați bugetul lunar înainte de cheltuieli?": "Budget_Planning",
        "Reușiți să economisiți bani lunar?": "Save_Money",
        "Ce anume vă împiedică să economisiți bani lunar?": "Savings_Obstacle",
        "Cât de frecvent faceți achiziții impulsive (neplanificate)?": "Impulse_Buying_Frequency",
        "Pe ce categorie sunt, de obicei, cheltuielile dumneavoastră impulsive?": "Impulse_Buying_Category",
        "Care este principalul motiv pentru cheltuielile impulsive?": "Impulse_Buying_Reason",
        "Ați folosit vreodată un credit sau o linie de împrumut?": "Credit_Usage",
        "Cum considerați nivelul actual al datoriilor dumneavoastră?": "Debt_Level",
        "Ați făcut investiții financiare până acum?": "Financial_Investments",
        "Cât de des analizați situația contului bancar (venituri și cheltuieli)?": "Bank_Account_Analysis_Frequency",
        "Cât timp utilizați, în general, următoarele tipuri de produse înainte de a le înlocui?   [Îmbrăcăminte]": "Product_Lifetime_Clothing",
        "Cât timp utilizați, în general, următoarele tipuri de produse înainte de a le înlocui?   [Gadget-uri și dispozitive tech (telefoane, laptopuri, tablete, console etc.)]": "Product_Lifetime_Tech",
        "Cât timp utilizați, în general, următoarele tipuri de produse înainte de a le înlocui?   [Electrocasnice (frigider, mașină de spălat etc.)]": "Product_Lifetime_Appliances",
        "Cât timp utilizați, în general, următoarele tipuri de produse înainte de a le înlocui?   [Autoturisme]": "Product_Lifetime_Cars",
        "Care este scopul principal al economiilor dumneavoastră?  \n  (Alegeți toate opțiunile care se aplică)": "Savings_Goal",
        "Cum distribuiți, în general, cheltuielile lunare?  \n(Maxim 3 categorii principale.)": "Expense_Distribution"
    }
    df.rename(columns=column_mapping, inplace=True)

    # Translate basic single-value fields
    basic_translation = {
        "Masculin": "Male",
        "Feminin": "Female",
        "Prefer să nu răspund": "Prefer not to say",
        "40-50": "41-50",
        "30-40": "31-40",
        "Necăsătorit/ă, fără copii": "Single, no children",
        "Necăsătorit/ă, cu copii": "Single, with children",
        "Într-o relație sau căsătorit/ă, cu copii.": "In a relationship/married with children",
        "Într-o relație (coabitare) sau căsătorit/ă, fără copii.": "In a relationship/married without children",
        "Într-o relație sau căsătorit/ă, fără copii.": "In a relationship/married without children",
        "Altul": "Another",
        "Încerc să găsesc un echilibru": "I try to find a balance",
        "Cheltuiesc mai mult decât câștig": "Spend more than I earn",
        "Sunt disciplinat/ă în economisire": "I am disciplined in saving",
        "Planific bugetul în detaliu": "Plan budget in detail",
        "Planific doar lucrurile esențiale": "Plan only essentials",
        "Nu planific deloc": "Don't plan at all",
        "Recompensă personală („merit acest lucru”)": "Self-reward",
        "Reduceri sau promoții": "Discounts or promotions",
        "Presiuni sociale („toți prietenii au acest lucru”)": "Social pressure",
        "Da, pentru cheltuieli neprevăzute dar inevitabile": "Yes, for unexpected expenses",
        "Inexistent": "Absent",
        "Scăzut": "Low",
        "Gestionabil": "Manageable",
        "Dificil de gestionat": "Difficult to manage",
        "Nu, dar sunt interesat/ă": "No, but interested",
        "Da, ocazional": "Yes, occasionally",
        "Da, regulat": "Yes, regularly",
        "Nu, nu sunt interesat/ă": "No, not interested",
        "Săptămânal": "Weekly",
        "Lunar": "Monthly",
        "Zilnic": "Daily",
        "Rar sau deloc": "Rarely or never",
        "Sub 6 luni": "<6 months",
        "Sub 50%": "<50%",
        "Peste 75%": ">75%",
        "6-12 luni": "6-12 months",
        "1-3 ani": "1-3 years",
        "3-5 ani": "3-5 years",
        "5-10 ani": "5-10 years",
        "10-15 ani": "10-15 years",
        "Peste 15 ani": ">15 years",
        "Sub 20": "<20",
        "Peste 50": ">50",
        "Nu am achizitionat": "Not purchased yet",
        "Uneori": "Sometimes",
        "Des": "Often",
        "Foarte rar": "Very rarely",
        "Da": "Yes",
        "Nu": "No",
        "12.000-16 000 RON": "12.000-16.000 RON",
        "8000-12.000 RON": "8.000-12.000 RON",
        "4000-8000 RON": "4.000-8.000 RON",
        "Peste 16.000 RON": ">16.000 RON",
        "Sub 4000 RON": "<4.000 RON",
    }
    df.replace(basic_translation, inplace=True)

    # Translate impulse buying fields
    impulse_map = {
        "Alimentație": "Food",
        "Haine sau produse de îngrijire personală": "Clothing or personal care products",
        "Electronice sau gadget-uri": "Electronics or gadgets",
        "Divertisment și timp liber": "Entertainment",
        "Altceva": "Other"
    }
    if "Impulse_Buying_Category" in df.columns:
        df["Impulse_Buying_Category"] = df["Impulse_Buying_Category"].replace(impulse_map)

    impulse_r_map = {"Altceva": "Other"}
    if "Impulse_Buying_Reason" in df.columns:
        df["Impulse_Buying_Reason"] = df["Impulse_Buying_Reason"].replace(impulse_r_map)

    # Translate and encode multi-option fields (multi-hot)
    multiple_val_map = {
        "Savings_Goal": {
            "Economii pentru achiziții majore (locuință, mașină)": "Major_Purchases",
            "Siguranță financiară pentru pensionare": "Retirement",
            "Fond de urgență": "Emergency_Fund",
            "Educația copiilor": "Child_Education",
            "Vacanțe sau cumpărături mari": "Vacation",
            "Altceva": "Other"
        },
        "Savings_Obstacle": {
            "Altceva": "Other",
            "Venitul este insuficient": "Insufficient_Income",
            "Alte cheltuieli urgente au prioritate": "Other_Expenses",
            "Nu consider economiile o prioritate": "Not_Priority"
        },
        "Expense_Distribution": {
            "Alimentație": "Food",
            "Locuință (chirie, utilități)": "Housing",
            "Transport": "Transport",
            "Divertisment și timp liber (iesiri cu prietenii, hobby-uri, excursii)": "Entertainment",
            "Sănătate (consultații medicale, medicamente, fizioterapie)": "Health",
            "Aspect personal (salon, cosmetice, haine, fitness)": "Personal_Care",
            "Cheltuieli generale pentru copii (îmbrăcăminte, activități extrașcolare)": "Child_Education",
            "Alte cheltuieli": "Other"
        },
        "Credit_Usage": {
            "Da, pentru cheltuieli esențiale (locuință, hrană)": "Essential_Needs",
            "Da, pentru cheltuieli mari (ex. vacanțe, electronice, autoturism, imobil etc.)": "Major_Purchases",
            "Da, pentru cheltuieli mari (ex. vacanțe, electronice)": "Major_Purchases",
            "Da, pentru cheltuieli neprevăzute dar inevitabile (ex. sănătate, reparații)": "Unexpected_Expenses",
            "Da, pentru nevoi personale (ex. evenimente speciale, educație)": "Personal_Needs",
            "Nu am folosit niciodată": "Never_Used"
        }
    }

    for col, translations in multiple_val_map.items():
        if col not in df.columns:
            continue

        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else "")
        df[col] = df[col].str.replace(r', (?=[A-ZĂÎȘȚÂ])', '; ', regex=True).str.strip()

        df[col] = df[col].apply(
            lambda cell: '; '.join(
                translations.get(part.strip(), part.strip()) for part in cell.split('; ')
            ) if cell else cell
        )

        col_text = df[col].copy()
        df[col] = df[col].str.split(r';\s*')

        for option in translations.values():
            dummy_col = f"{col}_{option}"
            df[dummy_col] = df[col].apply(
                lambda lst: int(option in lst if isinstance(lst, list) else 0)
            )

        df[col] = col_text


    return df


# Encode and prepare processed data for ML
def postprocess_data(df):
    try:
        # Map ordinal categories to integers
        ordinal_mappings = {
            'Impulse_Buying_Frequency': {
                'Very rarely': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Very often': 5
            },
            'Debt_Level': {
                'Absent': 1, 'Low': 2, 'Manageable': 3, 'Difficult to manage': 4, np.nan: 0, 'Unknown': 0
            },
            'Bank_Account_Analysis_Frequency': {
                'Rarely or never': 1, 'Monthly': 2, 'Weekly': 3, 'Daily': 4
            }
        }
        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        # One-hot encode categorical nominal values
        nominal_cols = [
            'Family_Status', 'Gender', 'Financial_Attitude', 'Budget_Planning',
            'Save_Money', 'Impulse_Buying_Category', 'Impulse_Buying_Reason', 'Financial_Investments', 'Savings_Obstacle'
        ]
        for col in nominal_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col).astype(int)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)

        # Ensure numeric types on key columns
        lifetime_cols = [
            'Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
            'Product_Lifetime_Appliances', 'Product_Lifetime_Cars'
        ]
        for col in lifetime_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median()).astype(int)

        numeric_cols = ['Age', 'Income_Category', 'Essential_Needs_Percentage'] + lifetime_cols
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median()).astype(int)

        return df
    except Exception as e:
        print(f"Critical preprocessing error: {str(e)}")
        return None


# Smooth fuzzy inputs (ranges, categories) to numeric values
def range_smoothing(df, age_column="Age", income_column="Income_Category", lifetime_columns=None,
                    essential_needs_column="Essential_Needs_Percentage", lifetime_func=None):
    if age_column in df.columns:
        df = replace_age_column(df, age_column)

    if income_column in df.columns:
        df = replace_income_category(df, income_column)

    if lifetime_columns:
        if lifetime_func is not None:
            df = replace_product_lifetime_columns(df, lifetime_columns, lifetime_func)
        else:
            df = replace_product_lifetime_columns(df, lifetime_columns, random_product_lifetime)

    if essential_needs_column in df.columns:
        df = replace_essential_needs(df, essential_needs_column)

    return df