import pandas as pd
from EDA.V1.data_loading import load_data
from EDA.V1.file_operations import select_save_directory
from FirstProcessing.file_operations import auto_adjust_column_width
import os

def trim_to_fixed_balanced_set(df, target_column="Behavior_Risk_Level", per_class=60):
    class_labels = df[target_column].unique()
    balanced_parts = [
        df[df[target_column] == label].sample(per_class, random_state=42)
        for label in class_labels
    ]
    balanced_df = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

def run_trimmer():
    df = load_data()
    if df is None:
        print("No dataset selected.")
        return

    balanced_df = trim_to_fixed_balanced_set(df)

    save_dir = select_save_directory()
    if not save_dir:
        print("No save folder selected.")
        return

    csv_path = os.path.join(save_dir, "TestSet_Balanced.csv")
    balanced_df.to_csv(csv_path, index=False)

    xlsx_path = os.path.join(save_dir, "TestSet_Balanced.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        balanced_df.to_excel(writer, index=False, sheet_name="TestSet")
        auto_adjust_column_width(writer, "TestSet")

    print(f"Saved 120-sample balanced dataset to:\n- {csv_path}\n- {xlsx_path}")

if __name__ == "__main__":
    run_trimmer()
