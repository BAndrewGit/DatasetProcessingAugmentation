import os
# Limit parallel processing CPU usage
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Grupuri de coloane pentru analiză
GROUP_MAP = {
    "demographic": ["Age", "Gender", "Income_Category"],
    "expenses": lambda cols: [c for c in cols if c.startswith("Expense_Distribution_")],
    "economy": lambda cols: [c for c in cols if c.startswith("Savings_Goal_") or c.startswith("Savings_Obstacle_")],
    "credit": lambda cols: [c for c in cols if c.startswith("Credit_Usage_") or c in ["Debt_Level", "Bank_Account_Analysis_Frequency"]],
    "behaviors": lambda cols: [c for c in cols if c.startswith("Financial_Attitude_") or c.startswith("Budget_Planning_")],
    "impulsivity": lambda cols: [c for c in cols if c.startswith("Impulse_Buying_")],
    "investments": lambda cols: [c for c in cols if c.startswith("Financial_Investments_")]
}

# Coloane de exclus
COLUMNS_TO_DROP = ['Confidence', 'Cluster', 'Auto_Label', 'Outlier', 'Behavior_Risk_Level']

# Target variable
TARGET = "Risk_Score"

# Subfolders pentru ploturi
PLOT_SUBFOLDERS = ["univariate", "bivariate", "vs_target", "pca", "clustering"]
PCA_VARIANCE_THRESHOLD = 0.80
CLUSTERING_K_RANGE = (2, 11)

# Plot settings
DPI = 300
