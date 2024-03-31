import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(filepath):
    data = pd.read_csv(filepath)

    y = data['mean_temp']  # Assuming 'mean_temp' is the target variable
    y_imputer = SimpleImputer(strategy='median')
    y = pd.DataFrame(y_imputer.fit_transform(y.values.reshape(-1, 1)), columns=['mean_temp'])

    X = data.drop('mean_temp', axis=1)
    X_imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(X_imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y.squeeze()
