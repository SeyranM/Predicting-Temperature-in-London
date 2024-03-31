import numpy as np
from matplotlib import pyplot as plt

from config import CFG
from src.preprocessing import preprocess_data
from src.train import train_models


def plot_feature_importance(model_name="Decision Tree"):
    X, y = preprocess_data(CFG.london_data_path)
    models_data_dict = train_models(X, y)
    models = models_data_dict["models"]
    X_train = models_data_dict["X_train"]
    y_train = models_data_dict["y_train"]
    model = models[model_name]
    model.fit(X_train, y_train)

    feature_names = X_train.columns.to_list()

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()


if __name__ == '__main__':

    plot_feature_importance(model_name="Decision Tree")