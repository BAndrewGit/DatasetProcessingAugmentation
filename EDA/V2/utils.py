import os
from tkinter import Tk, filedialog
from config import PLOT_SUBFOLDERS


def select_output_directory():
    root = Tk()
    root.withdraw()
    plots_dir = filedialog.askdirectory(title="Select folder to save plots", parent=root)
    root.destroy()

    if not plots_dir:
        raise ValueError("No folder selected for saving plots.")

    return plots_dir


def create_plot_directories(plots_dir):
    for subfolder in PLOT_SUBFOLDERS:
        os.makedirs(os.path.join(plots_dir, subfolder), exist_ok=True)
