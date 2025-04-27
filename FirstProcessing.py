import pandas as pd
import numpy as np
import re

import shap
from scipy.stats import truncnorm
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from tkinter import Tk, filedialog
from sklearn.metrics import silhouette_score, classification_report
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
from xgboost import XGBClassifier

# Set maximum CPU count for parallel processing
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Configuration for risk weights
CONFIG = {
    'weights': {
        'Budget_Planning_Plan budget in detail': 0.097,
        'Budget_Planning_Plan only essentials': 0.0805,
        'Age': 0.0782,
        'Family_Status_Single, no children': 0.0717,
        'Financial_Investments_Yes, regularly': 0.0705,
        'Gender_Male': 0.0682,
        'Impulse_Buying_Category_Food': 0.0682,
        'Family_Status_Another': 0.0647,
        'Impulse_Buying_Reason_Social pressure': 0.0629,
        'Impulse_Buying_Category_Other': 0.0611,
        'Gender_Female': 0.0594,
        'Impulse_Buying_Category_Entertainment': 0.0594,
        'Savings_Goal': 0.0547,
        'Savings_Obstacle_0001': 0.0529,
        'Savings_Obstacle_0010': 0.0506
    }
}

multi_value_cols = ["Savings_Goal", "Savings_Obstacle", "Expense_Distribution", "Credit_Usage"]


# Normalize and translate column names and values
def normalize_and_translate_data(df):
    df.columns = df.columns.str.strip()

    # Remove timestamp column if exists
    if df.columns[0].lower() in ['marcaj de timp', 'timestamp']:
        df.drop(columns=df.columns[0], inplace=True)

    # Rename columns using a single mapping pass
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

    # Replace repetitive values with English translations
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

    # Apply basic translations
    df.replace(basic_translation, inplace=True)

    # Translate impulse buying category values
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

    # Mapping for multi-value columns translation
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

    # Translate multi-value columns and encode binary strings (creăm coloane noi)
    for col, translations in multiple_val_map.items():
        if col not in df.columns:
            continue

        # TRADUCERE: păstrează coloana originală cu text tradus
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else "")
        df[col] = df[col].str.replace(r', (?=[A-ZĂÎȘȚÂ])', '; ', regex=True).str.strip()

        def translate_text(cell):
            if cell == "":
                return cell
            return '; '.join(translations.get(part.strip(), part.strip()) for part in cell.split('; '))
        df[col] = df[col].apply(translate_text)

        # ONE-HOT ENCODING pentru fiecare opțiune validă din dicționar
        options = list(translations.values())
        df[col] = df[col].astype(str).str.split(r';\s*')

        for option in options:
            new_col = f"{col}_{option}"
            df[new_col] = df[col].apply(lambda x: int(option in x if isinstance(x, list) else []))

        # Ștergem coloana originală dacă nu mai e necesară
        df.drop(columns=[col], inplace=True)

    return df


# Post-process data: encode ordinal/nominal values, convert types and impute missing data
def postprocess_data(df):
    try:
        # Ordinal Encoding numeric explicit
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

        # Nominal One-hot Encoding numeric explicit (fără dummy pentru NaN)
        nominal_cols = [
            'Family_Status', 'Gender', 'Financial_Attitude', 'Budget_Planning',
            'Save_Money', 'Impulse_Buying_Category', 'Impulse_Buying_Reason', 'Financial_Investments', 'Savings_Obstacle'
        ]
        for col in nominal_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col).astype(int)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)

        # Extindem coloanele multi-value în coloane binare (multi-label one-hot)
        multi_hot_map = {
            'Savings_Goal': ['Major_Purchases', 'Retirement', 'Emergency_Fund', 'Child_Education', 'Vacation', 'Other'],
            'Savings_Obstacle': ['Insufficient_Income', 'Other_Expenses', 'Not_Priority', 'Other'],
            'Expense_Distribution': ['Food', 'Housing', 'Transport', 'Entertainment', 'Health', 'Personal_Care',
                                     'Child_Education', 'Other'],
            'Credit_Usage': ['Essential_Needs', 'Major_Purchases', 'Unexpected_Expenses', 'Personal_Needs',
                             'Never_Used']
        }

        for col, values in multi_hot_map.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.split(r';\s*')
                for val in values:
                    df[f"{col}_{val}"] = df[col].apply(lambda x: int(val in x if isinstance(x, list) else []))
                df.drop(columns=[col], inplace=True)

        # Product Lifetime explicit numeric (în luni)
        lifetime_cols = [
            'Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
            'Product_Lifetime_Appliances', 'Product_Lifetime_Cars'
        ]
        for col in lifetime_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median()).astype(int)

        # Alte coloane strict numerice
        numeric_cols = ['Age', 'Income_Category', 'Essential_Needs_Percentage'] + lifetime_cols
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median()).astype(int)

        return df
    except Exception as e:
        print(f"Critical preprocessing error: {str(e)}")
        return None


# Calculate risk score using weighted conditions and clustering
def calculate_risk_clusters(df, cluster_range=(2, 8)):
    config = {
        'weights': {
            'Debt_Level': 0.25,
            'Impulse_Buying_Frequency': 0.15,
            'Essential_Needs_Percentage': -0.2,
            'Savings_Goal_Emergency_Fund': 0.1,
            'Bank_Account_Analysis_Frequency': -0.1
        }
    }

    # exact ca în calculate_risk_advanced
    def calculate_risk_score(df_local):
        scaler = RobustScaler()
        scaled = scaler.fit_transform(df_local[config['weights'].keys()])
        return pd.Series(np.dot(scaled, list(config['weights'].values())), index=df_local.index)

    df = df.copy()
    df['Risk_Score'] = calculate_risk_score(df)

    best_score = -1
    best_n_clusters = None
    best_model = None
    best_labels = None

    X = df[['Risk_Score']]

    for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
        model = BayesianGaussianMixture(n_components=n_clusters, max_iter=500, random_state=42)
        labels = model.fit_predict(X)

        try:
            score = silhouette_score(X, labels)
            print(f"Clusters={n_clusters}: Silhouette Score={score:.3f}")
        except Exception as e:
            print(f"Silhouette calculation failed for {n_clusters} clusters: {e}")
            continue

        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_model = model
            best_labels = labels

    if best_model is None:
        raise Exception("No valid clustering found.")

    print(f"\nBest clustering: {best_n_clusters} clusters with Silhouette Score={best_score:.3f}")

    df['Cluster'] = best_labels

    # Etichetare cluster sigur = cel cu scor mediu cel mai mic
    safe_cluster = df.groupby('Cluster')['Risk_Score'].mean().idxmin()
    df['Behavior_Risk_Level'] = np.where(df['Cluster'] == safe_cluster, 0, 1)

    print("\nFinal Risk distribution:")
    print(df['Behavior_Risk_Level'].value_counts())

    return df


def calculate_risk_advanced(df, gmm_clusters=4, confidence_threshold=0.95, iterations=4):
    config = {
        'weights': {
            'Debt_Level': 0.25,
            'Impulse_Buying_Frequency': 0.15,
            'Essential_Needs_Percentage': -0.2,
            'Savings_Goal_Emergency_Fund': 0.1,
            'Bank_Account_Analysis_Frequency': -0.1
        },
        'outlier_contamination': 0.05,
        'gmm_clusters': gmm_clusters,
        'knn_neighbors': 7
    }

    def calculate_risk_score(df_local):
        scaler = RobustScaler()
        scaled = scaler.fit_transform(df_local[config['weights'].keys()])
        return pd.Series(np.dot(scaled, list(config['weights'].values())), index=df_local.index)

    df = df.copy()
    df['Risk_Score'] = calculate_risk_score(df)

    iso = IsolationForest(contamination=config['outlier_contamination'], random_state=42)
    df['Outlier'] = np.where(iso.fit_predict(df[['Risk_Score']]) == -1, 1, 0)

    bgmm = BayesianGaussianMixture(n_components=config['gmm_clusters'],
                                   weight_concentration_prior=0.1,
                                   max_iter=500, random_state=42)
    df['Cluster'] = bgmm.fit_predict(df[['Risk_Score']])

    safe_cluster = np.argmin(bgmm.means_.flatten())
    df['Auto_Label'] = np.where(df['Cluster'] == safe_cluster, 0, 1)

    knn = KNeighborsClassifier(n_neighbors=config['knn_neighbors'])
    knn.fit(df[df['Outlier'] == 0][['Risk_Score']], df[df['Outlier'] == 0]['Auto_Label'])
    df.loc[df['Outlier'] == 1, 'Auto_Label'] = knn.predict(df[df['Outlier'] == 1][['Risk_Score']])

    df['Confidence'] = 1.0
    df['Behavior_Risk_Level'] = -1

    features = df.columns.difference(['Behavior_Risk_Level', 'Confidence', 'Auto_Label'])
    xgb = XGBClassifier(scale_pos_weight=1.5, max_depth=4, subsample=0.8, eval_metric='logloss', random_state=42)

    for iteration in range(iterations):
        train = df[df['Behavior_Risk_Level'] != -1]
        if len(train) < 10:
            train = df[df['Confidence'] > confidence_threshold]

        if len(train) < 10:
            print(f"Iteration {iteration + 1}: not enough high-confidence samples, stopping...")
            break

        print(f"Iteration {iteration + 1}: training on {len(train)} samples...")
        xgb.fit(train[features], train['Auto_Label'])

        probas = xgb.predict_proba(df.loc[df['Behavior_Risk_Level'] == -1, features])
        confidences = np.max(probas, axis=1)
        predictions = xgb.predict(df.loc[df['Behavior_Risk_Level'] == -1, features])

        high_confidence_indices = df.loc[df['Behavior_Risk_Level'] == -1].index[confidences >= confidence_threshold]

        df.loc[high_confidence_indices, 'Behavior_Risk_Level'] = predictions[confidences >= confidence_threshold]
        df.loc[high_confidence_indices, 'Confidence'] = confidences[confidences >= confidence_threshold]

        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(train[features])
        shap_summary = dict(zip(features, np.abs(shap_values).mean(axis=0)))
        print(f"Iteration {iteration + 1} SHAP Importances:", shap_summary)

        if len(high_confidence_indices) == 0:
            print(f"Iteration {iteration + 1}: no new high-confidence labels assigned, stopping...")
            break

    def apply_rules(row):
        score = 0
        if row['Income_Category'] >= 15000:
            score += 2
        if row['Debt_Level'] == 4:
            score -= 3
        if row['Debt_Level'] >= 2 and row['Income_Category'] < 12000:
             score -= 2
        if row.get('Financial_Investments_Yes, regularly', 0) == 1:
            score += 2
        if row.get('Budget_Planning_Plan budget in detail', 0) == 1:
            score += 1
        if row.get('Save_Money_Yes', 0) == 1:
            score += 1
        else:
            if row['Income_Category'] >= 5000:
                score -= 1
            else:
                score -= 0.5
        if row.get('Relationship_Status_In a relationship/married with children', 0) == 1:
            score += 1

        if score >= 2:
            return 0
        elif score <= -2:
            return 1
        return row['Behavior_Risk_Level']

    df['Behavior_Risk_Level'] = df.apply(apply_rules, axis=1)

    labeled = df[df['Behavior_Risk_Level'] != -1]
    unlabeled = df[df['Behavior_Risk_Level'] == -1]

    print(f"\nFinal labeled count: {len(labeled)}, Unlabeled (needs manual review): {len(unlabeled)}")

    # Scoring final
    if len(labeled) > 0:
        silhouette = silhouette_score(labeled[['Risk_Score']], labeled['Behavior_Risk_Level'])
        print(f"\nSilhouette Score (final labeled data): {silhouette:.2f}")

        print("\nClassification Report (final labeled data):")
        print(classification_report(labeled['Auto_Label'], labeled['Behavior_Risk_Level']))
    else:
        print("No labeled data available for final scoring.")

    return df


# Scale specified numeric columns using RobustScaler
def scale_numeric_columns(df, columns):
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


# Adjust Excel column widths based on content
def auto_adjust_column_width(writer, sheet_name):
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    for column_cells in worksheet.columns:
        max_length = 0
        column = column_cells[0].column_letter  # Get column letter
        for cell in column_cells:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except Exception as e:
                print(f"Error adjusting column: {e}")
        adjusted_width = max_length + 2
        worksheet.column_dimensions[column].width = adjusted_width


# Generate a random value from a truncated normal distribution
def truncated_normal(mean, std_dev, lower, upper):
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    return truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=1)[0]


# Generate random age based on string ranges (e.g., "30-40", ">50", "<20")
def random_age(value):
    value = str(value).strip()
    match = re.match(r"(\d+)\s*-\s*(\d+)", value)
    if match:
        lower, upper = map(int, match.groups())
        return np.random.randint(lower, upper + 1)
    match = re.match(r">\s*(\d+)", value)
    if match:
        num = int(match.group(1))
        if num >= 50:
            return np.random.randint(50, 71)
    match = re.match(r"<\s*(\d+)", value)
    if match:
        num = int(match.group(1))
        if num <= 20:
            return np.random.randint(18, 21)
    try:
        return int(value)
    except:
        return value

# Replace age column values using random_age
def replace_age_column(df, column_name="Age"):
    df[column_name] = df[column_name].apply(random_age)
    return df


# Generate random income based on string patterns and ranges
def random_income(value):
    value = str(value).replace(".", "").strip()
    match = re.match(r"(\d+)[^\d]+(\d+)", value)
    if match:
        lower, upper = map(int, match.groups())
        random_value = np.random.randint(lower // 100, (upper // 100) + 1) * 100
        return random_value
    match = re.match(r">\s*(\d+)", value)
    if match:
        lower = int(match.group(1))
        upper = lower + 10000
        random_value = np.random.randint(lower // 100, (upper // 100) + 1) * 100
        return random_value
    match = re.match(r"<\s*(\d+)", value)
    if match:
        upper = int(match.group(1))
        lower = 3000
        random_value = np.random.randint(lower // 100, (upper // 100) + 1) * 100
        return random_value
    try:
        return int(value) // 100 * 100  # Round to nearest 100
    except:
        return value

# Replace income category column values using random_income
def replace_income_category(df, column_name="Income_Category"):
    df[column_name] = df[column_name].apply(random_income)
    return df


# Generate random product lifetime based on input ranges
def random_product_lifetime(value, encoded=False):
    value = str(value).strip()
    if "Not purchased yet" in value:
        return 0 if encoded else value

    # Cazul în care valoarea e exprimată în luni, sub forma "< X months"
    match = re.match(r"<\s*(\d+)\s*months", value, re.IGNORECASE)
    if match:
        upper = int(match.group(1))
        lower = max(1, upper - 3)
        rand_val = np.random.randint(lower, upper)
        return rand_val if encoded else f"{rand_val} months"

    # Cazul în care valoarea este un interval în luni
    if "month" in value.lower():
        match = re.match(r"(\d+)\s*-\s*(\d+)\s*months", value, re.IGNORECASE)
        if match:
            lower, upper = map(int, match.groups())
            rand_val = np.random.randint(lower, upper + 1)
            return rand_val if encoded else f"{rand_val} months"
        match = re.match(r"(\d+)", value)
        if match:
            num = int(match.group(1))
            return num if encoded else f"{num} months"

    # Cazul în care valoarea este exprimată în ani
    if "year" in value.lower():
        match = re.match(r"(\d+)\s*-\s*(\d+)\s*years", value, re.IGNORECASE)
        if match:
            lower, upper = map(int, match.groups())
            rand_val = np.random.randint(lower, upper + 1)
            return rand_val * 12 if encoded else f"{rand_val} years"
        match = re.match(r">\s*(\d+)\s*years", value, re.IGNORECASE)
        if match:
            lower = int(match.group(1))
            upper = lower + 5
            rand_val = np.random.randint(lower, upper + 1)
            return rand_val * 12 if encoded else f"{rand_val} years"
        match = re.match(r"<\s*(\d+)\s*years", value, re.IGNORECASE)
        if match:
            upper = int(match.group(1))
            lower = max(1, upper - 5)
            rand_val = np.random.randint(lower, upper + 1)
            return rand_val * 12 if encoded else f"{rand_val} years"
        return value

    # Fallback
    return 0 if encoded else value


# Replace specified product lifetime columns using random_product_lifetime
def replace_product_lifetime_columns(df, columns, lifetime_func):
    for col in columns:
        df[col] = df[col].apply(lambda x: lifetime_func(x))
    return df


# Generate random essential needs percentage from input text
def random_essential_needs(value):
    if pd.isna(value) or str(value).strip().lower() == "nan":
        return np.nan

    str_value = str(value).strip().replace('%', '').replace(',', '.')
    min_val, max_val = None, None

    try:
        if '<' in str_value:
            min_val, max_val = 30, 50
        elif '>' in str_value:
            min_val, max_val = 75, 80
        elif '-' in str_value:
            parts = str_value.split('-')
            min_val = float(parts[0])
            max_val = float(parts[1])
        else:
            num_value = float(str_value)
            if num_value < 50:
                min_val, max_val = 30, 50
            elif 50 <= num_value <= 75:
                min_val, max_val = 50, 75
            else:
                min_val, max_val = 75, 80
    except ValueError:
        return np.nan

    print(f"[DEBUG] Processed value={value}, min={min_val}, max={max_val}")

    if min_val is None or max_val is None or np.isnan(min_val) or np.isnan(max_val):
        return np.nan

    if min_val > max_val:
        min_val, max_val = max_val, min_val
    elif min_val == max_val:
        return min_val

    random_num = np.random.uniform(min_val, max_val)
    rounded_value = np.round(random_num / 5) * 5
    rounded_value = max(min_val, min(rounded_value, max_val))

    return rounded_value

# Replace essential needs column using random_essential_needs
def replace_essential_needs(df, column_name="Essential_Needs_Percentage"):
    df[column_name] = df[column_name].apply(random_essential_needs)
    return df


# Apply range smoothing on age, income, product lifetime, and essential needs columns
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


# Save processed data to Excel and CSV files
def save_files(df):
    Tk().withdraw()  # Hide main Tkinter window
    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        title="Select save location for Excel"
    )

    if save_path:
        # Save as Excel
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Processed_Data')
            auto_adjust_column_width(writer, 'Processed_Data')
        print(f"Excel file saved at: {save_path}")

        # Save as CSV with same base filename
        filename_no_ext, _ = os.path.splitext(save_path)
        csv_save_path = filename_no_ext + ".csv"
        df.to_csv(csv_save_path, index=False, encoding='utf-8')
        print(f"CSV file saved at: {csv_save_path}")
    else:
        print("No save location selected.")


# Check and print NaN values in the DataFrame
def check_nan_values(df):
    nan_info = df.isna().sum()
    nan_columns = nan_info[nan_info > 0]

    if len(nan_columns) == 0:
        print("\n>>> INFO: No NaN values in data.")
        return

    print("\n>>> WARNING: Detected NaN values in the following columns:")
    for col, count in nan_columns.items():
        print(f"- Column '{col}': {count} NaN values")

    for col in nan_columns.index:
        nan_rows = df[df[col].isna()]
        print(f"\nFirst 5 rows with NaN in column '{col}':")
        print(nan_rows.head(5))


# Main function: file selection, data processing pipeline, and saving results
def main():
    Tk().withdraw()  # Ascundem fereastra principală Tkinter

    # Selectăm fișierul CSV de intrare
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select a CSV file to process"
    )
    if not file_path:
        print("No file selected.")
        return

    try:
        print(f"Loading file: {file_path}")
        df_raw = pd.read_csv(file_path, sep=",", quotechar='"', engine="python")

        print("\n>>> Normalizing and translating data...")
        df_norm = normalize_and_translate_data(df_raw)

        # Definim lista coloanelor de product lifetime
        lifetime_cols = [
            "Product_Lifetime_Clothing", "Product_Lifetime_Tech",
            "Product_Lifetime_Appliances", "Product_Lifetime_Cars"
        ]

        # Versiunea DECODIFIED (Excel): folosim coloanele originale cu text tradus
        df_decoded = df_norm.copy()
        # Eliminăm coloanele de encoding (care au sufixul '_encoded')
        encoded_cols = [col for col in df_decoded.columns if col.endswith('_encoded')]
        if encoded_cols:
            df_decoded.drop(columns=encoded_cols, inplace=True)

        # Aplicăm range smoothing pe versiunea decoded, folosind funcția care returnează text
        df_decoded = range_smoothing(
            df_decoded,
            age_column="Age",
            income_column="Income_Category",
            lifetime_columns=lifetime_cols,
            essential_needs_column="Essential_Needs_Percentage",
            lifetime_func=random_product_lifetime  # versiunea care returnează text
        )

        # Reordonăm coloanele conform ordinii dorite
        desired_column_order = [
            'Age', 'Family_Status', 'Gender', 'Income_Category', 'Essential_Needs_Percentage',
            'Financial_Attitude', 'Budget_Planning', 'Save_Money', 'Savings_Goal', 'Savings_Obstacle',
            'Expense_Distribution', 'Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
            'Product_Lifetime_Appliances', 'Product_Lifetime_Cars', 'Impulse_Buying_Frequency',
            'Impulse_Buying_Category', 'Impulse_Buying_Reason', 'Credit_Usage', 'Debt_Level',
            'Financial_Investments', 'Bank_Account_Analysis_Frequency', 'Behavior_Risk_Level'
        ]
        df_decoded = df_decoded[[col for col in desired_column_order if col in df_decoded.columns]]

        # Versiunea ENCODED (CSV): păstrăm doar coloanele de encoding
        df_encoded = df_norm.copy()
        # Ștergem coloanele originale de multi-value pentru care avem și versiuni encoded
        original_cols = multi_value_cols
        df_encoded.drop(columns=original_cols, inplace=True, errors='ignore')
        # Renumim coloanele cu sufixul '_encoded' pentru a elimina sufixul
        df_encoded.rename(columns=lambda x: x.replace('_encoded', ''), inplace=True)

        # Aplicăm range smoothing pe versiunea encoded, folosind funcția care returnează valori numerice (în luni)
        df_encoded = range_smoothing(
            df_encoded,
            age_column="Age",
            income_column="Income_Category",
            lifetime_columns=lifetime_cols,
            essential_needs_column="Essential_Needs_Percentage",
            lifetime_func=lambda x: random_product_lifetime(x, encoded=True)
        )

        print("\n>>> Post-processing data (encoded version)...")
        df_encoded = postprocess_data(df_encoded)
        if df_encoded is None:
            return

        # Scalăm coloanele numerice
        numeric_cols_to_scale = [
            'Age', 'Income_Category', 'Essential_Needs_Percentage',
            'Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
            'Product_Lifetime_Appliances', 'Product_Lifetime_Cars'
        ]
        df_encoded = scale_numeric_columns(df_encoded, numeric_cols_to_scale)

        print("\n>>> Calculating risk score...")
        df_encoded = calculate_risk_advanced(df_encoded) # Folosim funcția aleasa pentru a calcula riscul
        if df_encoded is None:
            return


        print("\nRisk distribution:")
        print(df_encoded['Behavior_Risk_Level'].value_counts(dropna=False))
        if len(df_encoded['Behavior_Risk_Level'].unique()) == 1:
            print("\nNot enough risk variation for analysis!")
            print("Possible solutions:")
            print("- Adjust CONFIG['risk_weights']")
            print("- Review input data")
            return

        # Pentru versiunea decoded, convertim nivelul de risc în etichete text
        df_decoded['Behavior_Risk_Level'] = df_encoded['Behavior_Risk_Level'].apply(
            lambda x: "Risky" if x == 1 else "Beneficial"
        )

        # Salvăm versiunea decoded ca fișier Excel
        Tk().withdraw()
        excel_save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Select save location for decoded Excel file"
        )
        if not excel_save_path:
            print("No save location selected for Excel.")
            return

        with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
            df_decoded.to_excel(writer, index=False, sheet_name='Decoded_Data')
            auto_adjust_column_width(writer, 'Decoded_Data')
        print(f"Decoded Excel file saved at: {excel_save_path}")

        # Salvăm versiunea encoded ca fișier CSV
        base_name, _ = os.path.splitext(excel_save_path)
        csv_save_path = base_name + "_encoded.csv"
        df_encoded.to_csv(csv_save_path, index=False, encoding='utf-8')
        print(f"Encoded CSV file saved at: {csv_save_path}")

        print("\nProcessing complete!")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
