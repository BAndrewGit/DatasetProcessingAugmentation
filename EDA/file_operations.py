import os
import json
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from FirstProcessing.file_operations import auto_adjust_column_width

# Open dialog to choose output folder
def select_save_directory():
    Tk().withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected if folder_selected else None

# Save plot to the selected directory
def save_plot(fig, save_dir, filename):
    if save_dir:
        path = os.path.join(save_dir, filename)
        fig.savefig(path)
        plt.close(fig)

# Save metrics to a JSON file
def save_metrics(results, overfit_report, feature_names):
    try:
        metrics = {}
        for name, res in results.items():
            metrics[name] = {
                'classification_report': res['classification_report'],
                'roc_auc': res['roc_auc'],
                'f1_score': res['f1_score'],
                'cv_f1_mean': res['cv_f1_mean'],
                'cv_f1_std': res['cv_f1_std'],
                'train_accuracy': overfit_report[name]['train_accuracy'],
                'test_accuracy': overfit_report[name]['test_accuracy'],
                'overfit_ratio': overfit_report[name]['overfit_ratio']
            }
            if name == 'LogisticRegression':
                coef_dict = {
                    feature: coef for feature, coef in zip(feature_names, res['model'].coef_[0])
                }
                sorted_coefs = dict(sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True))
                metrics[name]['coefficients'] = sorted_coefs

        with open('dataset_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print("Extended metrics have been saved to 'dataset_metrics.json'.")

    except Exception as e:
        print(f"Error saving metrics: {str(e)}")