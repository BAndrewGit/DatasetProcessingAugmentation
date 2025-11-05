from tkinter import Tk, filedialog
import pandas as pd

# Open file dialog and load CSV
def load_data():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    print("Select the dataset file...")
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], parent=root)
    root.destroy()
    return pd.read_csv(file_path) if file_path else None
