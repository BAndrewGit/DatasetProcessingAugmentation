from EDA.V1 import load_data
from config import COLUMNS_TO_DROP, GROUP_MAP

def load_and_prepare_data():
    df = load_data()
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    return df

def build_group_map(df):
    group_map = {}
    for group, col_def in GROUP_MAP.items():
        if callable(col_def):
            cols = col_def(df.columns)
        else:
            cols = [c for c in col_def if c in df.columns]
        group_map[group] = cols
    return group_map
