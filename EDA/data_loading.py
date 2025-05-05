from tkinter import Tk, filedialog
import pandas as pd

# Open file dialog and load CSV
def load_data():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return pd.read_csv(file_path) if file_path else None