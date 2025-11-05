from tkinter import Tk, filedialog
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from EDA import load_data

# Selectare folder pentru salvare ploturi
root = Tk()
root.withdraw()
PLOTS_DIR = filedialog.askdirectory(title="Select folder to save plots", parent=root)
root.destroy()

if not PLOTS_DIR:
    raise ValueError("No folder selected for saving plots.")

# Încărcare date
df = load_data()

# Drop coloane irelevante
df = df.drop(columns=['Confidence', 'Cluster', 'Auto_Label', 'Outlier', 'Behavior_Risk_Level'], errors='ignore')

# Definire grupuri de coloane
group_map = {
    "demographic": [c for c in df.columns if c in ["Age", "Gender", "Income_Category"]],
    "expenses": [c for c in df.columns if c.startswith("Expense_Distribution_")],
    "economy": [c for c in df.columns if c.startswith("Savings_Goal_") or c.startswith("Savings_Obstacle_")],
    "credit": [c for c in df.columns if
               c.startswith("Credit_Usage_") or c in ["Debt_Level", "Bank_Account_Analysis_Frequency"]],
    "behaviors": [c for c in df.columns if c.startswith("Financial_Attitude_") or c.startswith("Budget_Planning_")],
    "impulsivity": [c for c in df.columns if c.startswith("Impulse_Buying_")],
    "investments": [c for c in df.columns if c.startswith("Financial_Investments_")]
}

sns.set_theme(style="whitegrid")
target = "Risk_Score"

# Creare subfoldere
for subfolder in ["univariate", "bivariate", "vs_target"]:
    os.makedirs(os.path.join(PLOTS_DIR, subfolder), exist_ok=True)

# Univariate Analysis
print("Starting Univariate Analysis...")
for group, cols in group_map.items():
    for col in cols:
        nunique = df[col].nunique()
        if nunique <= 1:
            continue

        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(f"{group} — {col}", fontsize=14, y=0.98)

        # histogram + kde
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='steelblue')
        plt.title("Distribution", pad=10)

        # boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col], color='salmon', showfliers=True, flierprops={"marker": "o", "color": "black", "alpha": 0.6})
        plt.title("Boxplot", pad=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(PLOTS_DIR, "univariate", f"{group}_{col}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"- {group}/{col} saved")

# Bivariate Analysis
print("Starting Bivariate Analysis...")
for group, cols in group_map.items():
    numeric_cols = [c for c in cols if c in df.select_dtypes(include=np.number).columns]
    if len(numeric_cols) < 2:
        continue

    corr = df[numeric_cols].corr()
    high_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
        .head(3)
    )

    # heatmap global pe grup cu ajustări pentru nume lungi
    fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.8), max(8, len(numeric_cols) * 0.7)))

    # Scurtare nume coloane pentru heatmap
    short_names = [c.replace("Impulse_Buying_", "IB_").replace("Financial_", "Fin_")[:20] for c in numeric_cols]
    corr_renamed = corr.copy()
    corr_renamed.columns = short_names
    corr_renamed.index = short_names

    sns.heatmap(corr_renamed, annot=True, cmap="coolwarm", fmt=".2f",
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"Correlation Heatmap — {group}", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bivariate", f"{group}_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"- {group} heatmap saved")

    # scatter pentru top 3 perechi corelate
    for (f1, f2), val in high_pairs.items():
        fig = plt.figure(figsize=(7, 6))
        sns.regplot(x=df[f1], y=df[f2], scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
        plt.title(f"{group}: {f1} vs {f2}\n(r={val:.2f})", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "bivariate", f"{group}_{f1}_vs_{f2}.png"), dpi=300, bbox_inches='tight')
        plt.close()

# Analysis vs Target
print("Starting Analysis vs Target...")
all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in all_numeric_cols if c != target]

corr_with_target = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
top5 = corr_with_target.head(5)

# Scatter plots top 5
for col in top5.index:
    fig = plt.figure(figsize=(7, 6))
    sns.regplot(x=df[col], y=df[target], scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    plt.title(f"{col} vs {target}\n(corr={corr_with_target[col]:.2f})", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "vs_target", f"{col}_vs_{target}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"- {col} vs {target} saved")

# Barplot feature importance
fig = plt.figure(figsize=(9, 6))
ax = sns.barplot(x=top5.values, y=top5.index, orient="h", hue=top5.index, palette="Blues_r", legend=False)
ax.set_title("Top 5 Features correlated with Risk_Score", fontsize=14, pad=15)
ax.set_xlabel("Absolute Correlation", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "vs_target", "feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()
print("- Feature importance saved")

print(f"\nAll plots saved to: {PLOTS_DIR}")
