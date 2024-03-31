import mlflow

from config import CFG
from src.train import train_models
from src.evaluate import evaluate_model
from src.visualizations.prediction_visualization import plot_results
from src.preprocessing import preprocess_data


def execute_pipeline():
    mlflow.set_experiment("Weather_Prediction")

    X, y = preprocess_data(CFG.london_data_path)

    models_data_dict = train_models(X, y)
    models = models_data_dict["models"]
    X_train = models_data_dict["X_train"]
    X_test = models_data_dict["X_test"]
    y_train = models_data_dict["y_train"]
    y_test = models_data_dict["y_test"]

    rmse_values = {}
    for name, model in models.items():
        rmse = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        rmse_values[name] = rmse

    plot_results(rmse_values)

    experiment_id = mlflow.get_experiment_by_name("Weather_Prediction").experiment_id
    experiment_results = mlflow.search_runs(experiment_ids=[experiment_id])
    return experiment_results


if __name__ == "__main__":
    experiment_results_ = execute_pipeline()
