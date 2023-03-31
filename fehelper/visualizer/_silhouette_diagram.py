import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.metrics import silhouette_samples, silhouette_score


def silhouette_diagram(X_train, kprot_runs):
    """
    Plot the silhouette diagram using k-prototypes clustering runs results.

    :param X_train: Training set of X
    :param kprot_runs: the number of runs of k-prototypes clustering
    :return:
    """
    silhouette_scores = [silhouette_score(X_train, model.labels_)
                         for model in kprot_runs[1:]]

    plt.figure(figsize=(15, 20))
    for k in range(2, 10):
        plt.subplot(4, 2, k - 1)

        y_pred = kprot_runs[k - 1].labels_
        silhouette_coefficients = silhouette_samples(X_train, y_pred)

        padding = len(X_train) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()

            color = cm.Spectral(i / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if k in (2, 4, 6, 8):
            plt.ylabel("Cluster")

        if k in (8, 9):
            plt.xlabel("Silhouette Coefficient")

        plt.axvline(x=silhouette_scores[k - 1], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)

    plt.show()