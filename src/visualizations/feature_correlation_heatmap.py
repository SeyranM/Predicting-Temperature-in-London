import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import CFG


def plot_correlation_heatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(CFG.london_data_path)
    plot_correlation_heatmap(data=df)
