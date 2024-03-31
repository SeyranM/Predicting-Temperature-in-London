import matplotlib.pyplot as plt


def plot_results(rmse_values):
    plt.bar(rmse_values.keys(), rmse_values.values())
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('Model Performance Comparison')
    plt.show()
