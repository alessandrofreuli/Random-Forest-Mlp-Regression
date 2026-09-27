"""Plotting utilities."""
import numpy as np
import matplotlib.pyplot as plt

from . import config


def plot_feature_scatter(data, target: str = config.TARGET):
    """Scatter plot of every numeric feature against the target."""
    for f in data.select_dtypes([np.number]):
        plt.scatter(data[f], data[target], alpha=0.1, s=5)
        plt.xlabel(f)
        plt.ylabel(target)
        plt.show()


def plot_histograms(data, figsize=(12, 8), bins=50):
    """Histograms of all numeric columns."""
    data.hist(bins=bins, figsize=figsize)
    plt.show()


def plot_model_comparison(y_test, preds_rf, preds_mlp):
    """Scatter comparison of actual values, RF predictions and MLP predictions."""
    x_idx = np.arange(len(y_test))
    plt.scatter(x_idx, y_test, alpha=0.1, s=5, c="green", label="Actual")
    plt.scatter(x_idx, preds_rf, alpha=0.1, s=5, c="blue", label="Random Forest")
    plt.scatter(x_idx, preds_mlp, alpha=0.1, s=5, c="red", label="MLP")
    plt.xlabel("Sample")
    plt.ylabel(config.TARGET)
    plt.legend()
    plt.show()
