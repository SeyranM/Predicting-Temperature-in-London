from pathlib import Path


class CFG:

    parent_dir = Path(__file__).parent
    data_dir = parent_dir.joinpath("data")
    raw_data_dir = data_dir.joinpath("raw_data")
    london_data_path = raw_data_dir.joinpath("london_weather.csv")

    model_names = [
        'Linear Regression',
        'Decision Tree',
        'Random Forest'
    ]
