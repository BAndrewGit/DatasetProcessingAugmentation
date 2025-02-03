import pandas as pd
import numpy as np
import re
from scipy.stats import truncnorm
import openpyxl
import os
from tkinter import Tk, filedialog

def normalize_and_translate_data(df):
    df.columns = df.columns.str.strip()

    # 1) Eliminăm coloana de timestamp, dacă există
    if df.columns[0].lower() in ['marcaj de timp', 'timestamp']:
        df.drop(columns=df.columns[0], inplace=True)


    # 2) Redenumire coloane (single pass)
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

    # 3) Dicționar de traduceri single-value (non-multi-choice)
    #    - Valorile repetitive (Masculin->Male, etc.)
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
        "Altul": "Another",
        "Venitul este insuficient": "Income is insufficient",
        "Alte cheltuieli urgente au prioritate": "Other urgent expenses take priority",
        "Nu consider economiile o prioritate": "I don't consider savings a priority",
        "Recompensă personală („merit acest lucru”)": "Self-reward",
        "Reduceri sau promoții": "Discounts or promotions",
        "Presiuni sociale („toți prietenii au acest lucru”)": "Social pressure",
        "Da, pentru cheltuieli neprevăzute dar inevitabile": "Yes, for unexpected expenses",
        "Inexistent": "None",
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


    # Aplicăm traducerile single-value
    df.replace(basic_translation, inplace=True)

    # Traducem `Impulse_Buying_Category`
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

    # Traducem `Savings_Obstacle`
    obstacle_map = {"Altceva": "Other"}
    if "Savings_Obstacle" in df.columns:
        df["Savings_Obstacle"] = df["Savings_Obstacle"].replace(obstacle_map)

    # 4) Dicționare pentru coloanele multi-value (fiecare coloană are map-ul său)
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

    credit_map = {
        "Da, pentru cheltuieli esențiale (locuință, hrană)": "Credit_Essential_Needs",
        "Da, pentru cheltuieli mari (ex. vacanțe, electronice)": "Credit_Major_Purchases",
        "Da, pentru cheltuieli neprevăzute dar inevitabile (ex. sănătate, reparații)": "Credit_Unexpected_Expenses",
        "Da, pentru nevoi personale (ex. evenimente speciale, educație)": "Credit_Personal_Needs",
        "Nu am folosit niciodată": "Credit_Never_Used"
    }

    # 5) Dicționar cu ce coloane (multi-value) procesezi la one-hot
    one_hot_columns = {
        "Savings_Goal": [
            "Savings_Goal_Emergency_Fund",
            "Savings_Goal_Major_Purchases",
            "Savings_Goal_Child_Education",
            "Savings_Goal_Vacation",
            "Savings_Goal_Retirement",
            "Savings_Goal_Other"
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

    # 6) One-hot maps (folosite la translate) - cheie = numele coloanei, valoare = map dedicat
    one_hot_maps = {
        "Savings_Goal": savings_map,
        "Expense_Distribution": expense_map,
        "Credit_Usage": credit_map
    }

    # Funcție de mapping + split pe care o aplicăm în bucla de mai jos
    def sub_translate(col_name, text):
        if pd.isnull(text):
            return text
        comps = text.split(';')
        current_map = one_hot_maps.get(col_name, {})
        mapped = [current_map.get(c.strip(), c.strip()) for c in comps]
        return ';'.join(mapped)

    # 7) Procesăm fiecare coloană multi-value definită
    for col, new_cols in one_hot_columns.items():
        if col in df.columns:
            # Curățăm spațiile la ';'
            df[col] = df[col].str.replace(r'\s*;\s*', ';', regex=True).str.strip()

            # Traducem fiecare element
            df[col] = df[col].apply(lambda x: sub_translate(col, x))

            # get_dummies
            dummies = df[col].str.get_dummies(';')
            dummies = dummies.reindex(columns=new_cols, fill_value=0)

            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    return df


def auto_adjust_column_width(writer, sheet_name):
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    for column_cells in worksheet.columns:
        max_length = 0
        column = column_cells[0].column_letter  # Get column name
        for cell in column_cells:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except Exception as e:
                print(f"Error adjusting column: {e}")
        adjusted_width = max_length + 2
        worksheet.column_dimensions[column].width = adjusted_width

def truncated_normal(mean, std_dev, lower, upper):
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    return truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=1)[0]

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
        return int(value) // 100 * 100  # Rotunjim la cel mai apropiat multiplu de 100
    except:
        return value

def random_product_lifetime(value):
    value = str(value).strip()
    if "Not purchased yet" in value:
        return value
    match = re.match(r"<\s*(\d+)\s*months", value, re.IGNORECASE)
    if match:
        upper = int(match.group(1))
        lower = max(1, upper - 3)  # Generează între [upper-3, upper)
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

def random_essential_needs(value):
    if pd.isna(value) or str(value).strip().lower() == "nan":
        return np.nan

    str_value = str(value).strip().replace('%', '').replace(',', '.')
    min_val, max_val = None, None

    try:
        if '<' in str_value:
            # Case: "<50%" → [30, 50)
            min_val, max_val = 30, 50
        elif '>' in str_value:
            # Case: ">75%" → (75, 80]
            min_val, max_val = 75, 80
        elif '-' in str_value:
            # Case: "50-75%"
            parts = str_value.split('-')
            min_val = float(parts[0])
            max_val = float(parts[1])
        else:
            # Numeric handling (e.g., "45", "60%")
            num_value = float(str_value)
            if num_value < 50:
                min_val, max_val = 30, 50
            elif 50 <= num_value <= 75:
                min_val, max_val = 50, 75
            else:
                min_val, max_val = 75, 80
    except ValueError:
        return np.nan  # Handle invalid splits or conversions

    # Debug print to check the interval (now safe)
    print(f"[DEBUG] Processed value={value}, min={min_val}, max={max_val}")

    # Handle invalid ranges (e.g., min > max)
    if min_val is None or max_val is None or np.isnan(min_val) or np.isnan(max_val):
        return np.nan  # Invalid configuration

    # Ensure valid range (swap if needed)
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    elif min_val == max_val:
        return min_val  # No randomness needed

    # Generate random number in [min_val, max_val]
    random_num = np.random.uniform(min_val, max_val)

    # Round to nearest 5
    rounded_value = np.round(random_num / 5) * 5

    # Clamp to [min_val, max_val] (inclusive)
    rounded_value = max(min_val, min(rounded_value, max_val))

    return rounded_value

def replace_age_column(df, column_name="Age"):
    df[column_name] = df[column_name].apply(random_age)
    return df

def replace_income_category(df, column_name="Income_Category"):
    df[column_name] = df[column_name].apply(random_income)
    return df

def replace_product_lifetime_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(random_product_lifetime)
    return df

def replace_essential_needs(df, column_name="Essential_Needs_Percentage"):
    df[column_name] = df[column_name].apply(random_essential_needs)
    return df

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

def save_files(df):
    # Select location to save the processed file (Excel)
    Tk().withdraw()
    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        title="Select save location for Excel"
    )

    if save_path:
        # 1) Salvăm Excel
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Processed_Data')
            auto_adjust_column_width(writer, 'Processed_Data')
        print(f"Excel file saved at: {save_path}")

        # 2) Salvăm CSV (cu aceeași bază de fișier, dar extensia .csv)
        filename_no_ext, _ = os.path.splitext(save_path)  # eliminăm .xlsx
        csv_save_path = filename_no_ext + ".csv"
        df.to_csv(csv_save_path, index=False, encoding='utf-8')
        print(f"CSV file saved at: {csv_save_path}")

    else:
        print("No save location selected.")


def main():
    # Ascundem fereastra principală Tkinter
    Tk().withdraw()

    # Selectăm fișierul de intrare
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select a CSV file to process"
    )

    if not file_path:
        print("No file selected.")
        return

    try:
        test_values = ["<50%", "50-75%", ">75%", "45", "80%", "50-abc", "NaN", "invalid"]
        for val in test_values:
            result = random_essential_needs(val)
            print(f"Input: {val} => Output: {result}")
        # IMPORTANT: Specificăm sep="," și quotechar='"'
        print(f"Loading file: {file_path}")
        df = pd.read_csv(file_path, sep=",", quotechar='"', engine="python")

        print("\n>>> Normalizing and translating data...")
        df_processed = normalize_and_translate_data(df)

        print("\n>>> Applying range smoothing...")
        df_processed = range_smoothing(df_processed, age_column="Age", income_column="Income_Category",
                                       lifetime_columns=["Product_Lifetime_Clothing", "Product_Lifetime_Tech", "Product_Lifetime_Appliances", "Product_Lifetime_Cars"])

        # Debug pentru coloane după procesare
        print("\n>>> DEBUG: Columns AFTER processing:")
        print(df_processed.columns)

        print("\n>>> Saving processed data...")
        save_files(df_processed)

        print("\nProcessing complete!")

    except Exception as e:
        print(f"An error occurred during processing: {e}")



if __name__ == "__main__":
    main()
