from numba.core.types import containers
import pyod
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.knn import KNN
from sklearn import metrics as met

"""
Notes from the lab attendant
confusion matrix - predictions on the top, true values vertical
TP,FN
FP,TN
P=TN/TN+FN
"""

CONTAMINATION_RATE = 0.1
TRAINING_SAMPLES = 400
TEST_SAMPLES = 100


def main() -> None:
    train_points, test_points, train_values, test_values = pyod.utils.generate_data(
        TRAINING_SAMPLES, TEST_SAMPLES, 2, CONTAMINATION_RATE, random_state=1
    )
    # print(len(train_points), len(test_points), len(train_values), len(test_values))
    # print(type(train_points), type(test_points), type(train_values), type(test_values))
    # ex1
    plt.scatter(
        np.concat(
            (
                train_points[:, 0],
                test_points[:, 0],
            )
        ),
        np.concat(
            (
                train_points[:, 1],
                test_points[:, 1],
            )
        ),
        None,
        np.concat(
            (
                train_values,
                (test_values + 1) * 2,
            )
        ),
    )
    plt.show()
    # ex2
    model = KNN(contamination=CONTAMINATION_RATE)
    model.fit(train_points, train_values)

    tr_pred = model.predict(train_points)
    te_pred = model.predict(test_points)

    # print(tr_pred)

    # TN FP
    # FN TP
    tr_mat = met.confusion_matrix(train_values, tr_pred, labels=[0, 1])
    te_mat = met.confusion_matrix(test_values, te_pred, labels=[0, 1])

    print("training confusion matrix:")
    print(tr_mat)
    print("training accuracy:", met.balanced_accuracy_score(train_values, tr_pred))

    print("[" + "-" * 20 + "]")

    print("testing confusion matrix:")
    print(te_mat)
    print("training accuracy:", met.balanced_accuracy_score(test_values, te_pred))

    # ROC curve
    te_scores = model.decision_function(test_points)
    # print(te_scores, type(te_scores)) # >1???
    fpr, tpr, thresholds = met.roc_curve(test_values, te_scores)

    plt.plot(fpr, tpr, label="KNN")
    plt.plot([0, 1], [0, 1], "k--")
    plt.show()

    for c in [0.05, 0.1, 0.2]:
        model = KNN(contamination=c)
        model.fit(train_points, train_values)
        te_pred = model.predict(test_points)

        print("[" + "-" * 20 + "]")

        print(f"contamination: {c}")
        print("confusion matrix:")
        print(met.confusion_matrix(test_values, te_pred, labels=[0, 1]))
        print("accuracy:", met.balanced_accuracy_score(test_values, te_pred))

    # ex3
    #
    #

    SAMPLES = 1000
    train_points, test_points, train_values, test_values = pyod.utils.generate_data(
        SAMPLES, 0, 1, CONTAMINATION_RATE, random_state=1
    )

    # print(train_points)
    mean = np.mean(train_points)
    dev = np.std(test_points)
    z_score = np.abs((train_points.flatten() - mean) / dev)
    thr = np.quantile(z_score, CONTAMINATION_RATE)

    pred = (z_score >= thr).astype(int)

    print("[" + "-" * 20 + "]")
    print("ex3 accuracy:", met.balanced_accuracy_score(train_values, pred))

    # ex 4
    #
    CONTAMINATION_SAMPLES = int(SAMPLES * CONTAMINATION_RATE)
    N_SAMPLES = SAMPLES - CONTAMINATION_SAMPLES

    D = 3
    np.random.seed(1)

    mu = np.array([1.0, -2.0, 3.0])
    sigma = np.array(
        [
            [2, 2, 3],
            [2, 5, 6],
            [3, 6, 8],
        ]
    )
    l = np.linalg.cholesky(sigma)
    # print(l)
    normal = np.random.normal(size=(N_SAMPLES, D)) @ l.T + mu
    # print(normal)

    # ex4 does not specify conatmination as a requirement but it is implied F

    contamination = (
        np.random.random(size=(CONTAMINATION_SAMPLES, D)) @ l.T + mu
    )  # source is comming from another distribtions as per the course (or at least as one of the causes of anomalies mentioned in the course)

    normal = np.vstack((normal, contamination))

    truth = np.concatenate((np.zeros(N_SAMPLES), np.ones(CONTAMINATION_SAMPLES)))

    permut = np.random.permutation(SAMPLES)

    normal = normal[permut]
    truth = truth[permut]

    mean = np.mean(normal, axis=0)
    dev = np.std(normal, axis=0)
    # print(dev, mean)
    z_mat = np.abs((normal - mean) / dev)
    z_score = np.linalg.norm(z_mat, axis=1)

    thr = np.quantile(z_score, 1 - CONTAMINATION_RATE)
    # print(z_score)
    #
    pred = (z_score >= thr).astype(int)

    print("[" + "-" * 20 + "]")
    print(
        "ex4 accuracy:", met.balanced_accuracy_score(train_values, pred)
    )  # why so low idk
