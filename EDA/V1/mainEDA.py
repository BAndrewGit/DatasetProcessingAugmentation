import pandas as pd
from sklearn.model_selection import train_test_split
from EDA.V1.data_loading import load_data
from EDA.V1.preprocessing import preprocess_encoded_data
from EDA.V1.model_training import train_models, evaluate_overfitting, drop_artificial_features
from EDA.V1.visualization import visualize_data
from EDA.V1.file_operations import save_metrics

def remove_exact_duplicates(train_df, test_df):
    print(f"Train before removing duplicates: {len(train_df)}")
    print(f"Test reference size: {len(test_df)}")

    # Sort & align columns
    train_df = train_df.sort_values(by=train_df.columns.tolist()).reset_index(drop=True)
    test_df = test_df.sort_values(by=test_df.columns.tolist()).reset_index(drop=True)

    train_df["_source"] = "train"
    test_df["_source"] = "test"
    combined = pd.concat([train_df, test_df], ignore_index=True)

    # Drop perfect duplicates
    dedup = combined.drop_duplicates(subset=train_df.columns.difference(["_source"]).tolist(), keep=False)
    filtered_train = dedup[dedup["_source"] == "train"].drop(columns=["_source"])

    print(f"Train after removing exact matches from test: {len(filtered_train)}")
    return filtered_train

def main():
    print("Select test dataset (original or full data)...")
    df_test = load_data()
    if df_test is None:
        print("No file selected. Aborting.")
        return

    print("Now select train dataset (synthetic) or Cancel to auto-split...")
    df_train = load_data()

    if df_train is None:
        print("One dataset selected. Proceeding with split...")
        df_test = preprocess_encoded_data(df_test)
        df_test = drop_artificial_features(df_test)

        if df_test is None:
            return

        X = df_test.drop(columns=["Behavior_Risk_Level"])
        y = df_test["Behavior_Risk_Level"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
    else:
        print("Two datasets selected. Using test + train directly.")
        df_test = preprocess_encoded_data(df_test)
        df_train = preprocess_encoded_data(df_train)

        df_test = drop_artificial_features(df_test)
        df_train = drop_artificial_features(df_train)

        if df_test is None or df_train is None:
            return

        # Elimina rândurile duplicate exacte
        df_train = remove_exact_duplicates(df_train, df_test)

        X_train = df_train.drop(columns=["Behavior_Risk_Level"])
        y_train = df_train["Behavior_Risk_Level"]
        X_test = df_test.drop(columns=["Behavior_Risk_Level"])
        y_test = df_test["Behavior_Risk_Level"]

    # Train models
    df_combined = pd.concat([X_train, y_train], axis=1)
    models, results, *_ = train_models(X_train, X_test, y_train, y_test)

    if models is None:
        return

    models_results = {name: (model, X_test, y_test) for name, model in models.items()}
    df_viz, save_dir = visualize_data(pd.concat([X_test, y_test], axis=1), models_results)

    overfit_report = evaluate_overfitting(models, X_train, X_test, y_train, y_test)
    save_metrics(results, overfit_report, feature_names=X_train.columns, save_dir=save_dir)

if __name__ == "__main__":
    main()
