import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
import numpy as np


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, model_name)

        return rmse
