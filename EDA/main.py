from EDA.data_loading import load_data
from EDA.preprocessing import preprocess_encoded_data
from EDA.model_training import train_models, evaluate_overfitting
from EDA.visualization import visualize_data
from EDA.file_operations import save_metrics

def main():
    df = load_data()
    if df is None:
        return

    # Fill missing values
    df = preprocess_encoded_data(df)
    if df is None:
        return

    # Train models and get results
    models, results, X_train, X_test, y_train, y_test = train_models(df)
    if models is None:
        return

    # Visualize data and model performance
    models_results = {
        name: (model, X_test, y_test) for name, model in models.items()
    }

    df_viz = visualize_data(df, models_results=models_results)

    # Evaluate overfitting and save all metrics
    overfit_report = evaluate_overfitting(models, X_train, X_test, y_train, y_test)
    save_metrics(results, overfit_report, feature_names=X_train.columns)

if __name__ == "__main__":
    main()