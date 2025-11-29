# task5.py
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42
REDUCED_PATH = Path("cifar_pca_200.npz")


def load_reduced_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find reduced dataset at {path}. Run task4.py first."
        )
    data = np.load(path)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def main():
    X_train, y_train, X_test, y_test = load_reduced_data(REDUCED_PATH)

    params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
    }

    clf = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_SEED),
        param_grid=params,
        cv=3,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("Best params:", clf.best_params_)

    best_tree = clf.best_estimator_
    y_pred = best_tree.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (decision tree): {test_acc:.4f}")


if __name__ == "__main__":
    main()
