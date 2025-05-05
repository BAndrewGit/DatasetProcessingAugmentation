import numpy as np
import re
import pandas as pd
from scipy.stats import truncnorm


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