from loguru import logger
from matplotlib import pyplot as plt

from config import CFG
from src.preprocessing import preprocess_data
from src.train import train_models


def plot_actual_vs_predicted(model_name="Linear Regression"):
    if model_name not in CFG.model_names:
        logger.error(f"You provided invalid model name: {model_name}.")
        logger.error(f"Please provide a model name from this list: {CFG.model_names}")
        return
    X, y = preprocess_data(CFG.london_data_path)

    models_data_dict = train_models(X, y)
    models = models_data_dict["models"]
    X_train = models_data_dict["X_train"]
    y_train = models_data_dict["y_train"]
    X_test = models_data_dict["X_test"]
    y_test = models_data_dict["y_test"]
    model = models[model_name]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.show()


if __name__ == '__main__':
    plot_actual_vs_predicted(model_name="Linear Regression")
