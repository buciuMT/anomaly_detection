from typing import cast
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import pyod
from pyod.models import combination as pyodcomb
from pyod.models import knn as pyodknn
from pyod.models import lof as pyodlof
from pyod.utils.utility import standardizer
from scipy.io import loadmat
from sklearn.datasets import make_blobs
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def linearmodel(x: np.ndarray) -> np.ndarray:
    return x * 2 + 4  # xkcd 221


def plane_model(x: np.ndarray) -> np.ndarray:
    return 2 * x[0] + 3 * x[1] + 4


def lscores(data: np.ndarray) -> np.ndarray:
    # diag(x(x^t x)^-1 x^t)
    return data @ np.linalg.inv(data.T @ data) @ data.T


def ex1():
    SAMPLES_SIZE = 400
    CONTAMINATION = 0.1
    CONTAMINATION_SIZE = int(SAMPLES_SIZE * CONTAMINATION)
    distribution_v = ((1, 1), (1, 20))
    noise_v = ((0.1, 1), (0.1, 20))
    fig, axs = plt.subplots(2, 2)
    for d, d_kind, plot_half in zip(distribution_v, ("lo", "hi"), axs):
        for n, n_kind, ax in zip(noise_v, ("lo", "hi"), plot_half):
            ax.set_title(f"distribution:{d_kind} noise:{n_kind}")
            x = np.random.normal(*d, SAMPLES_SIZE)  # xkcd 221
            line = np.array(min(x), max(x))
            noise = np.random.normal(*n, SAMPLES_SIZE)
            y = linearmodel(x) + noise
            samples = np.vstack((x, y)).swapaxes(0, 1)
            h_diag = lscores(samples).diagonal()
            print(f"h_diag:{h_diag}\n\n" + ("==" * 20) + "\n\n")
            print(h_diag.sum())
            diag_norm = h_diag / np.max(h_diag)
            ax.scatter(x, y, c=diag_norm)
            ax.plot(line, linearmodel(line), color="b", linestyle="dashed")

    plt.show()

    distribution_v = ((1, 1), (1, 20))
    noise_v = ((0.1, 1), (0.1, 20))
    fig = plt.figure()

    for d, d_kind, plot_half in zip(distribution_v, ("lo", "hi"), (0, 1)):
        for n, n_kind, quart in zip(noise_v, ("lo", "hi"), (0, 1)):
            ax = fig.add_subplot(2, 2, plot_half * 2 + quart + 1, projection="3d")
            ax.set_title(f"distribution:{d_kind} noise:{n_kind}")
            x = np.random.normal(*d, (2, SAMPLES_SIZE))  # xkcd 221
            noise = np.random.normal(*n, SAMPLES_SIZE)
            y = plane_model(x) + noise
            samples = np.vstack((x, y)).swapaxes(0, 1)
            print(samples.shape)  # .swapaxes(0, 1)
            h_diag = lscores(samples).diagonal()
            print(f"h_diag, projection3d:{h_diag}\n\n" + ("==" * 20) + "\n\n")
            print(h_diag.sum())
            diag_norm = h_diag / np.max(h_diag)
            ax.scatter(x[0], x[1], y, c=diag_norm)
    plt.show()


def ex2():
    SAMPLES_CNT = 400
    TEST_SAMPLES_CNT = 200
    CONTAMINATION = 0.1
    x_train: NDArray
    y_train: NDArray
    x_test: NDArray
    y_test: NDArray
    (x_train, x_test, y_train, y_test) = cast(
        tuple[NDArray, NDArray, NDArray, NDArray],
        pyod.utils.data.generate_data_clusters(
            SAMPLES_CNT, TEST_SAMPLES_CNT, 2, 2, CONTAMINATION
        ),
    )
    fig = plt.figure()
    for idx, n_neighbors in enumerate([5, 10, 20, 35]):
        ground_train = fig.add_subplot(4, 4, idx * 4 + 1)
        predicted_train = fig.add_subplot(4, 4, idx * 4 + 2)
        ground_test = fig.add_subplot(4, 4, idx * 4 + 3)
        predicted_test = fig.add_subplot(4, 4, idx * 4 + 4)
        knn = pyodknn.KNN(CONTAMINATION, n_neighbors=n_neighbors)
        knn.fit(x_train, y_train)
        ground_train.scatter(x_train[:, 0], x_train[:, 1], c=y_train.astype(np.int_))
        ground_test.scatter(x_test[:, 0], x_test[:, 1], c=y_test.astype(np.int_))
        pred_train = knn.predict(x_train)
        pred_test = knn.predict(x_test)
        predicted_train.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c=(pred_train * 2 + y_train).round().astype(np.int_),
        )
        predicted_test.scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=(pred_test * 2 + y_test).round().astype(np.int_),
        )
        print(f"train BA{balanced_accuracy_score(y_train, pred_train)}")
        print(f"test BA {balanced_accuracy_score(y_test, pred_test)}")
    plt.show()


def ex3():
    DENSE_CLUSTER_SIZE = 200
    SPARSE_CLUSTER_SIZE = 100
    CONTAMINATION = 0.07
    samples, _ = cast(
        tuple[NDArray, NDArray],
        make_blobs(
            n_samples=[DENSE_CLUSTER_SIZE, SPARSE_CLUSTER_SIZE],
            centers=[(-10, -10), (10, 10)],
            cluster_std=[2, 6],
            random_state=RANDOM_SEED,
        ),
    )
    neighbors_v = (5, 10, 20, 35)
    fig, axs = plt.subplots(len(neighbors_v), 2)
    for n_neighbors, (knn_ax, lof_ax) in zip(neighbors_v, axs):
        knn_ax.set_title(f"KNN n_neighbors:{n_neighbors}")
        lof_ax.set_title(f"LOF n_neighbors:{n_neighbors}")
        knn = pyodknn.KNN(CONTAMINATION, n_neighbors=n_neighbors)
        lof = pyodlof.LOF(n_neighbors=n_neighbors, contamination=CONTAMINATION)
        knn.fit(samples)
        lof.fit(samples)
        knn_pred = knn.predict(samples)
        lof_pred = lof.predict(samples)
        knn_ax.scatter(samples[:, 0], samples[:, 1], c=knn_pred)
        lof_ax.scatter(samples[:, 0], samples[:, 1], c=lof_pred)

    fig.suptitle(f"contamination:{CONTAMINATION}")
    plt.show()


def ex4():
    TEST_SIZE = 0.3
    data = loadmat("cardio.mat")  # for lab attendant, the file is gitignored
    samples: NDArray = data["X"]
    labels: NDArray = data["y"].ravel()
    CONTAMINATION = float(np.mean(labels))
    x_train, x_test, y_train, y_test = cast(
        tuple[NDArray, NDArray, NDArray, NDArray],
        train_test_split(
            samples,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=labels,
        ),
    )
    scaler = StandardScaler().fit(x_train)
    x_train = cast(np.ndarray, scaler.transform(x_train))
    x_test = cast(np.ndarray, scaler.transform(x_test))
    neighbors_v = range(30, 121, 10)

    for model_name, model_cls in (("KNN", pyodknn.KNN), ("LOF", pyodlof.LOF)):
        train_scores = np.zeros((x_train.shape[0], len(neighbors_v)))
        test_scores = np.zeros((x_test.shape[0], len(neighbors_v)))
        for idx, n_neighbors in enumerate(neighbors_v):
            model = model_cls(contamination=CONTAMINATION, n_neighbors=n_neighbors)
            model.fit(x_train)
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)
            train_ba = balanced_accuracy_score(y_train, train_pred)
            # print(train_ba.shape)
            test_ba = balanced_accuracy_score(y_test, test_pred)
            print(
                f"{model_name} n_neighbors:{n_neighbors} train BA:{train_ba} test BA:{test_ba}"
            )
            train_scores[:, idx] = model.decision_function(x_train)
            test_scores[:, idx] = model.decision_function(x_test)

        train_scores, test_scores = standardizer(train_scores, test_scores)
        # print(train_scores)
        for strategy_name, strategy in (
            ("average", pyodcomb.average),
            ("maximization", pyodcomb.maximization),
        ):
            train_combined = strategy(train_scores)
            test_combined = strategy(test_scores)
            threshold = np.quantile(train_combined, 1 - CONTAMINATION)
            train_pred = (train_combined > threshold).astype(np.int_)
            test_pred = (test_combined > threshold).astype(np.int_)
            train_ba = balanced_accuracy_score(y_train, train_pred)
            test_ba = balanced_accuracy_score(y_test, test_pred)
            print(
                f"{model_name} {strategy_name} threshold:{threshold} train BA:{train_ba} test BA:{test_ba}"
            )


# chosen by fair dice rol
# guranteed to be random
RANDOM_SEED = 4


def main() -> None:
    np.random.seed(RANDOM_SEED)
    # ex1()
    # ex2()
    # ex3()
    ex4()
