import traceback
import pandas as pd
import numpy as np
import re
from scipy.stats import truncnorm
from sklearn.cluster import KMeans
from tkinter import Tk, filedialog
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os

# Set maximum CPU count for parallel processing
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Configuration for risk weights
CONFIG = {
    'risk_weights': [0.50, 0.20, 0.15, 0.15]
}


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
    savings_map = {
        "Economii pentru achiziții majore (locuință, mașină)": "Savings_Goal_Major_Purchases",
        "Siguranță financiară pentru pensionare": "Savings_Goal_Retirement",
        "Fond de urgență": "Savings_Goal_Emergency_Fund",
        "Educația copiilor": "Savings_Goal_Child_Education",
        "Vacanțe sau cumpărături mari": "Savings_Goal_Vacation",
        "Altceva": "Savings_Goal_Other"
    }

    expense_map = {
        "Alimentație": "Expense_Distribution_Food",
        "Locuință (chirie, utilități)": "Expense_Distribution_Housing",
        "Transport": "Expense_Distribution_Transport",
        "Divertisment și timp liber (iesiri cu prietenii, hobby-uri, excursii)": "Expense_Distribution_Entertainment",
        "Sănătate (consultații medicale, medicamente, fizioterapie)": "Expense_Distribution_Health",
        "Aspect personal (salon, cosmetice, haine, fitness)": "Expense_Distribution_Personal_Care",
        "Cheltuieli generale pentru copii (îmbrăcăminte, activități extrașcolare)": "Expense_Distribution_Child_Education",
        "Alte cheltuieli": "Expense_Distribution_Other"
    }

    savings_obstacle_map = {
        "Altceva": "Savings_Obstacle_Other",
        "Venitul este insuficient": "Savings_Obstacle_Insufficient_Income",
        "Alte cheltuieli urgente au prioritate": "Savings_Obstacle_Other_Expenses",
        "Nu consider economiile o prioritate": "Savings_Obstacle_Not_Priority"
    }

    credit_map = {
        "Da, pentru cheltuieli esențiale (locuință, hrană)": "Credit_Essential_Needs",
        "Da, pentru cheltuieli mari (ex. vacanțe, electronice)": "Credit_Major_Purchases",
        "Da, pentru cheltuieli neprevăzute dar inevitabile (ex. sănătate, reparații)": "Credit_Unexpected_Expenses",
        "Da, pentru nevoi personale (ex. evenimente speciale, educație)": "Credit_Personal_Needs",
        "Nu am folosit niciodată": "Credit_Never_Used"
    }

    # Columns for one-hot encoding of multi-value fields
    one_hot_columns = {
        "Savings_Goal": [
            "Savings_Goal_Emergency_Fund",
            "Savings_Goal_Major_Purchases",
            "Savings_Goal_Child_Education",
            "Savings_Goal_Vacation",
            "Savings_Goal_Retirement",
            "Savings_Goal_Other"
        ],
        "Savings_Obstacle": [
            "Savings_Obstacle_Insufficient_Income",
            "Savings_Obstacle_Other_Expenses",
            "Savings_Obstacle_Not_Priority",
            "Savings_Obstacle_Other"
        ],
        "Expense_Distribution": [
            "Expense_Distribution_Food",
            "Expense_Distribution_Housing",
            "Expense_Distribution_Transport",
            "Expense_Distribution_Entertainment",
            "Expense_Distribution_Health",
            "Expense_Distribution_Personal_Care",
            "Expense_Distribution_Child_Education",
            "Expense_Distribution_Other"
        ],
        "Credit_Usage": [
            "Credit_Essential_Needs",
            "Credit_Major_Purchases",
            "Credit_Unexpected_Expenses",
            "Credit_Personal_Needs",
            "Credit_Never_Used"
        ]
    }

    # Mapping for one-hot translation per column
    one_hot_maps = {
        "Savings_Goal": savings_map,
        "Savings_Obstacle": savings_obstacle_map,
        "Expense_Distribution": expense_map,
        "Credit_Usage": credit_map
    }

    # Helper function to translate multi-value entries
    def sub_translate(col_name, text):
        if pd.isnull(text):
            return text
        comps = text.split(';')
        current_map = one_hot_maps.get(col_name, {})
        mapped = [current_map.get(c.strip(), c.strip()) for c in comps]
        return ';'.join(mapped)

    # Process each multi-value column: clean, translate and one-hot encode
    for col, new_cols in one_hot_columns.items():
        if col in df.columns:
            # Clean spaces around delimiters
            df[col] = df[col].str.replace(r'\s*;\s*', ';', regex=True).str.strip()
            # Translate each element
            df[col] = df[col].apply(lambda x: sub_translate(col, x))
            # One-hot encode the values
            dummies = df[col].str.get_dummies(';')
            dummies = dummies.reindex(columns=new_cols, fill_value=0)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    return df


# Post-process data: encode ordinal/nominal values, convert types and impute missing data
def postprocess_data(df):
    try:
        # Map ordinal categorical columns to numeric values
        ordinal_mappings = {
            'Impulse_Buying_Frequency': {
                'Very rarely': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Very often': 5
            },
            'Debt_Level': {
                'Difficult to manage': 4, 'Manageable': 3, 'Low': 2, 'Absent': 1, np.nan: 0, 'Unknown': 0
            },
            'Bank_Account_Analysis_Frequency': {
                'Rarely or never': 1, 'Monthly': 2, 'Weekly': 3, 'Daily': 4
            }
        }

        for col, mapping in ordinal_mappings.items():
            df[col] = df[col].map(mapping).fillna(0).astype(int)

        # One-hot encode nominal categorical columns with less than 10 unique values
        nominal_cols = [
            'Family_Status', 'Gender', 'Financial_Attitude', 'Budget_Planning',
            'Save_Money', 'Impulse_Buying_Category', 'Impulse_Buying_Reason',
            'Financial_Investments', 'Savings_Obstacle'
        ]
        nominal_cols = [col for col in nominal_cols if col in df.columns]

        for col in nominal_cols:
            if df[col].nunique() < 10:
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)

        # Convert Credit_* columns to binary (1 or 0)
        credit_cols = [c for c in df.columns if c.startswith('Credit_')]
        for col in credit_cols:
            df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)

        # Convert remaining object columns to numeric; drop if conversion fails
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except:
                df.drop(columns=[col], inplace=True)

        # Impute missing values: median for numeric, mode for non-numeric
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        return df

    except Exception as e:
        print(f"Critical preprocessing error: {str(e)}")
        traceback.print_exc()
        return None


# Calculate risk score using weighted conditions and clustering
def calculate_risk(df, threshold=None):
    try:
        numeric_cols = [
            'Income_Category', 'Essential_Needs_Percentage',
            'Expense_Distribution_Entertainment', 'Debt_Level',
            'Savings_Goal_Emergency_Fund'
        ]
        # Convert specified columns to numeric
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute risk score as weighted sum of conditions
        conditions = [
            ((df['Income_Category'] < 5000) & (df['Essential_Needs_Percentage'] < 45)) * CONFIG['risk_weights'][0],
            ((df['Income_Category'] > 7500) & (df['Essential_Needs_Percentage'] > 60)) * -CONFIG['risk_weights'][0],
            (df['Expense_Distribution_Entertainment'] > 25) * CONFIG['risk_weights'][1],
            (df['Debt_Level'] >= 2) * CONFIG['risk_weights'][2],
            (df['Savings_Goal_Emergency_Fund'] == 0) * CONFIG['risk_weights'][3]
        ]
        df['Risk_Score'] = np.sum(conditions, axis=0)

        # Scale risk score to normalize differences in scale
        scaler = MinMaxScaler()
        df['Risk_Score_scaled'] = scaler.fit_transform(df[['Risk_Score']])

        # Determine dynamic threshold using KMeans clustering if not provided
        if threshold is None:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            df['Cluster_Label'] = kmeans.fit_predict(df[['Risk_Score_scaled']])
            risky_cluster = df.groupby('Cluster_Label')['Risk_Score_scaled'].mean().idxmax()
            threshold_scaled = df[df['Cluster_Label'] == risky_cluster]['Risk_Score_scaled'].min()
            print(f"Dynamic threshold (KMeans) [scaled]: {threshold_scaled:.2f}")
            threshold = scaler.inverse_transform([[threshold_scaled]])[0][0]

        # Label behavior based on risk score and threshold
        df['Behavior_Risk_Level'] = np.where(df['Risk_Score'] >= threshold, 1, 0)

        # Remove auxiliary columns
        df.drop(columns=['Risk_Score', 'Risk_Score_scaled', 'Cluster_Label'], inplace=True)
        return df, threshold
    except Exception as e:
        print(f"Error calculating risk score: {e}")
        return None, None


# Progressive risk calculation with iterative clustering and confidence labeling
def calculate_risk_progressive(df, threshold=None, distance_threshold=0.1, max_iter=3):
    try:
        # Convert relevant columns to numeric
        numeric_cols = [
            'Income_Category', 'Essential_Needs_Percentage',
            'Expense_Distribution_Entertainment', 'Debt_Level',
            'Savings_Goal_Emergency_Fund'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute risk score as weighted sum of conditions
        conditions = [
            ((df['Income_Category'] < 5000) & (df['Essential_Needs_Percentage'] < 45)) * CONFIG['risk_weights'][0],
            ((df['Income_Category'] > 7500) & (df['Essential_Needs_Percentage'] > 60)) * -CONFIG['risk_weights'][0],
            (df['Expense_Distribution_Entertainment'] > 25) * CONFIG['risk_weights'][1],
            (df['Debt_Level'] >= 2) * CONFIG['risk_weights'][2],
            (df['Savings_Goal_Emergency_Fund'] == 0) * CONFIG['risk_weights'][3]
        ]
        df['Risk_Score'] = np.sum(conditions, axis=0)

        # Scale risk score for clustering
        scaler = MinMaxScaler()
        df['Risk_Score_scaled'] = scaler.fit_transform(df[['Risk_Score']])

        # Determine initial threshold with clustering if not provided
        if threshold is None:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            df['Cluster_Label'] = kmeans.fit_predict(df[['Risk_Score_scaled']])
            risky_cluster = df.groupby('Cluster_Label')['Risk_Score_scaled'].mean().idxmax()
            threshold_scaled = df[df['Cluster_Label'] == risky_cluster]['Risk_Score_scaled'].min()
            threshold = scaler.inverse_transform([[threshold_scaled]])[0][0]
            print(f"Initial dynamic threshold (scaled): {threshold_scaled:.2f} -> threshold: {threshold:.2f}")

        # Initialize risk labels as uncertain (-1)
        df['Behavior_Risk_Level'] = -1

        # Iterative labeling based on confidence (distance to cluster centroid)
        for i in range(max_iter):
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            df['Cluster_Label'] = kmeans.fit_predict(df[['Risk_Score_scaled']])
            centroids = kmeans.cluster_centers_

            # Calculate distance from each point to its cluster centroid
            df['Distance_to_Centroid'] = df.apply(
                lambda row: abs(row['Risk_Score_scaled'] - centroids[int(row['Cluster_Label'])][0]), axis=1)

            # Label points with high confidence (distance within threshold)
            high_confidence = df['Distance_to_Centroid'] <= distance_threshold
            df.loc[high_confidence, 'Behavior_Risk_Level'] = np.where(
                df.loc[high_confidence, 'Risk_Score'] >= threshold, 1, 0
            )

            num_uncertain = np.sum(df['Behavior_Risk_Level'] == -1)
            print(f"Iteration {i + 1}: {num_uncertain} instances remain uncertain.")
            if num_uncertain < 0.05 * len(df):
                break

        # Remove auxiliary columns
        df.drop(columns=['Risk_Score', 'Risk_Score_scaled', 'Cluster_Label', 'Distance_to_Centroid'], inplace=True)
        return df, threshold

    except Exception as e:
        print(f"Error in progressive risk calculation: {e}")
        return None, None


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
        return int(value) // 100 * 100  # Round to nearest hundred
    except:
        return value

# Replace income category column values using random_income
def replace_income_category(df, column_name="Income_Category"):
    df[column_name] = df[column_name].apply(random_income)
    return df


# Generate random product lifetime based on input ranges
def random_product_lifetime(value):
    value = str(value).strip()
    if "Not purchased yet" in value:
        return value
    match = re.match(r"<\s*(\d+)\s*months", value, re.IGNORECASE)
    if match:
        upper = int(match.group(1))
        lower = max(1, upper - 3)
        rand_val = np.random.randint(lower, upper)
        return f"{rand_val} months"
    if "month" in value.lower():
        match = re.match(r"(\d+)\s*-\s*(\d+)\s*months", value, re.IGNORECASE)
        if match:
            lower, upper = map(int, match.groups())
            rand_val = np.random.randint(lower, upper + 1)
            return f"{rand_val} months"
        match = re.match(r"(\d+)", value)
        if match:
            return f"{match.group(1)} months"
    if "year" in value.lower():
        match = re.match(r"(\d+)\s*-\s*(\d+)\s*years", value, re.IGNORECASE)
        if match:
            lower, upper = map(int, match.groups())
            rand_val = np.random.randint(lower, upper + 1)
            return f"{rand_val} years"
        match = re.match(r">\s*(\d+)\s*years", value, re.IGNORECASE)
        if match:
            lower = int(match.group(1))
            upper = lower + 5
            rand_val = np.random.randint(lower, upper + 1)
            return f"{rand_val} years"
        match = re.match(r"<\s*(\d+)\s*years", value, re.IGNORECASE)
        if match:
            upper = int(match.group(1))
            lower = max(1, upper - 5)
            rand_val = np.random.randint(lower, upper + 1)
            return f"{rand_val} years"
        return value

# Replace specified product lifetime columns using random_product_lifetime
def replace_product_lifetime_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(random_product_lifetime)
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
                    essential_needs_column="Essential_Needs_Percentage"):

    if age_column in df.columns:
        df = replace_age_column(df, age_column)

    if income_column in df.columns:
        df = replace_income_category(df, income_column)

    if lifetime_columns:
        df = replace_product_lifetime_columns(df, lifetime_columns)

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
    Tk().withdraw()  # Hide Tkinter main window

    # Select input CSV file
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select a CSV file to process"
    )

    if not file_path:
        print("No file selected.")
        return

    try:
        # Optional test for random_essential_needs function
        test_values = ["<50%", "50-75%", ">75%", "45", "80%", "50-abc", "NaN", "invalid"]
        for val in test_values:
            result = random_essential_needs(val)
            print(f"Input: {val} => Output: {result}")

        print(f"Loading file: {file_path}")
        df = pd.read_csv(file_path, sep=",", quotechar='"', engine="python")

        print("\n>>> Normalizing and translating data...")
        df = normalize_and_translate_data(df)

        print("\n>>> Applying range smoothing...")
        df = range_smoothing(
            df,
            age_column="Age",
            income_column="Income_Category",
            lifetime_columns=["Product_Lifetime_Clothing", "Product_Lifetime_Tech",
                              "Product_Lifetime_Appliances", "Product_Lifetime_Cars"]
        )

        # Keep a copy before post-processing for decoded Excel file
        df_decoded = df.copy()

        print("\n>>> Post-processing data...")
        df = postprocess_data(df)
        if df is None:
            return

        # Scale numeric columns for encoded CSV version
        numeric_cols_to_scale = [
            'Age', 'Income_Category', 'Essential_Needs_Percentage',
            'Expense_Distribution_Entertainment', 'Debt_Level', 'Savings_Goal_Emergency_Fund'
        ]
        df = scale_numeric_columns(df, numeric_cols_to_scale)

        print("\n>>> Calculating risk score...")
        df, risk_threshold = calculate_risk_progressive(df, threshold=None)
        if df is None:
            return

        print(f"\n🔹 Global threshold: {risk_threshold:.2f}")
        with open("global_risk_threshold.txt", "w") as f:
            f.write(f"{risk_threshold:.2f}")

        print("\nRisk distribution:")
        print(df['Behavior_Risk_Level'].value_counts(dropna=False))
        if len(df['Behavior_Risk_Level'].unique()) == 1:
            print("\nNot enough risk variation for analysis!")
            print("Possible solutions:")
            print("- Adjust CONFIG['risk_weights']")
            print("- Review input data")
            return

        # Decode risk levels for Excel file: 0 -> "Beneficial", 1 -> "Risky"
        def decode_risk_level(x):
            return "Risky" if x == 1 else "Beneficial"

        df_decoded['Behavior_Risk_Level'] = df['Behavior_Risk_Level'].apply(decode_risk_level)

        # Select save location for decoded Excel file
        Tk().withdraw()
        excel_save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Select save location for decoded Excel file"
        )
        if not excel_save_path:
            print("No save location selected for Excel.")
            return

        # Save decoded Excel file
        with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
            df_decoded.to_excel(writer, index=False, sheet_name='Decoded_Data')
            auto_adjust_column_width(writer, 'Decoded_Data')
        print(f"Decoded Excel file saved at: {excel_save_path}")

        # Save encoded CSV file with a modified filename
        base_name, _ = os.path.splitext(excel_save_path)
        csv_save_path = base_name + "_encoded.csv"
        df.to_csv(csv_save_path, index=False, encoding='utf-8')
        print(f"Encoded CSV file saved at: {csv_save_path}")

        print("\nProcessing complete!")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
