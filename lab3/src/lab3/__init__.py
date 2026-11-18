from typing import Any, Tuple, cast
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import pyod
from pyod.models import combination as pyodcomb
from pyod.models import knn as pyodknn
from pyod.models import lof as pyodlof
from pyod.utils.utility import standardizer
from scipy.io import loadmat
from scipy.stats import alpha
from sklearn.datasets import make_blobs
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA


def ex1():
    x_train, _ = make_blobs(
        n_samples=500, centers=[(0, 0)], cluster_std=1.0, n_features=2, random_state=4
    )
    x_train = cast(np.ndarray, x_train)
    # print(x_train.shap
    NUM_PROJ = 5
    NUM_BINS = 20

    projs = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=NUM_PROJ)
    projs /= np.linalg.norm(projs, axis=1, keepdims=True)
    # print(projs)
    y_train = x_train @ projs.T
    counts = np.array(
        [
            np.histogram(y_train[:, j], bins=NUM_BINS, range=(-5, 5))[0]
            for j in range(NUM_PROJ)
        ]
    ).T
    print(counts)

    probs = counts / len(x_train)
    x_test = np.random.uniform(low=-3, high=3, size=(500, 2))
    y_test = x_test @ projs.T
    # print(y_test.shape)
    binw = 10 / NUM_BINS  # range=(-5, 5)
    binidx = np.floor((y_test - (-5)) / binw).astype(int)

    pidx = np.arange(NUM_PROJ)
    # print(pidx)
    prob_test = probs[binidx, pidx]

    anomaly_scores = np.mean(prob_test, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(
        x_train[:, 0],
        x_train[:, 1],
    )
    axes[0].grid(alpha=0.3)
    scatter = axes[1].scatter(
        x_test[:, 0], x_test[:, 1], c=anomaly_scores, cmap="coolwarm", s=25
    )
    plt.colorbar(scatter, ax=axes[1])
    plt.show()


def ex2():
    fig, axes = plt.subplots(3, 3)

    x_train, _ = make_blobs(
        n_samples=1000,
        centers=[(10, 0), (0, 10)],
        n_features=2,
    )
    x_train = cast(np.ndarray, x_train)
    # print(x_train.shape)
    x_test = np.random.uniform(-5, 15, size=(1000, 2))

    for iter, neur in enumerate([2, 20, 100]):
        iff = IForest(contamination=0.02)
        iff.fit(x_train)
        anomaly_scores = iff.decision_function(x_test)
        axes[0][iter].scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=anomaly_scores,
        )

        iff = DIF(contamination=0.02, hidden_neurons=[neur])
        iff.fit(x_train)
        anomaly_scores = iff.decision_function(x_test)
        axes[1][iter].scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=anomaly_scores,
        )

        iff = LODA(contamination=0.02, n_bins=neur)
        iff.fit(x_train)
        anomaly_scores = iff.decision_function(x_test)
        axes[2][iter].scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=anomaly_scores,
        )
    plt.show()  # not scaled???

    fig, axes = plt.subplots(3, 3, subplot_kw={"projection": "3d"})

    x_train, _ = make_blobs(
        n_samples=1000,
        centers=[(0, 10, 0), (10, 0, 10)],
        n_features=3,
    )
    x_train = cast(np.ndarray, x_train)
    # print(x_train.shape)

    x_test = np.random.uniform(-5, 15, size=(1000, 3))

    for iter, neur in enumerate([2, 20, 100]):
        iff = IForest(contamination=0.02)
        iff.fit(x_train)
        anomaly_scores = iff.decision_function(x_test)
        axes[0][iter].scatter(
            x_test[:, 0],
            x_test[:, 1],
            x_test[:, 2],
            c=anomaly_scores,
            alpha=0.5,
        )

        iff = DIF(contamination=0.02, hidden_neurons=[neur])
        iff.fit(x_train)
        anomaly_scores = iff.decision_function(x_test)
        axes[1][iter].scatter(
            x_test[:, 0],
            x_test[:, 1],
            x_test[:, 2],
            c=anomaly_scores,
            alpha=0.5,
        )

        iff = LODA(contamination=0.02, n_bins=neur)
        iff.fit(x_train)
        anomaly_scores = iff.decision_function(x_test)
        axes[2][iter].scatter(
            x_test[:, 0],
            x_test[:, 1],
            x_test[:, 2],
            c=anomaly_scores,
            alpha=0.5,
        )
    plt.show()  # not scaled???


# chosen by fair dice rol
# guranteed to be random
RANDOM_SEED = 4


def main() -> None:
    np.random.seed(RANDOM_SEED)
    # ex1()
    ex2()
    # ex3()
    # ex4()
