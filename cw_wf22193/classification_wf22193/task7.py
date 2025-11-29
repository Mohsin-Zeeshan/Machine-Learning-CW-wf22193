# task7.py
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

RANDOM_SEED = 42
REDUCED_PATH = Path("cifar_pca_200.npz")
MAX_TRAIN_SAMPLES = 20000  # to keep SVM training reasonable


def load_reduced_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find reduced dataset at {path}. Run task4.py first."
        )
    data = np.load(path)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def main():
    X_train, y_train, X_test, y_test = load_reduced_data(REDUCED_PATH)

    if X_train.shape[0] > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(X_train.shape[0], size=MAX_TRAIN_SAMPLES, replace=False)
        X_train_cv = X_train[idx]
        y_train_cv = y_train[idx]
    else:
        X_train_cv = X_train
        y_train_cv = y_train

    param_grid = [
        {
            "kernel": ["linear"],
            "C": [0.1, 1, 10],
        },
        {
            "kernel": ["rbf"],
            "C": [1, 10],
            "gamma": ["scale", 0.01],
        },
    ]

    svm = SVC(random_state=RANDOM_SEED)

    clf = GridSearchCV(
        svm,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
    )

    clf.fit(X_train_cv, y_train_cv)

    print("Best params (SVM):", clf.best_params_)

    best_svm = clf.best_estimator_
    y_pred = best_svm.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (SVM): {test_acc:.4f}")


if __name__ == "__main__":
    main()
