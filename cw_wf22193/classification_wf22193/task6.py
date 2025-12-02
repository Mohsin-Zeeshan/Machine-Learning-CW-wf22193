# task6.py
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

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

#Hyperparameters for random forest
    param_dist = {
        "criterion": ["gini", "entropy"],
        "n_estimators": [100, 200],
        "max_depth": [None, 20, 40],
        "max_features": ["sqrt", 0.5],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 2],
    }

#Initialise random forest with parallelisation
    rf = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    search.fit(X_train, y_train)

    print("Best params (RF):", search.best_params_)

#Evaluate best model on test set
    best_rf = search.best_estimator_
    y_pred = best_rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (random forest): {test_acc:.4f}")


if __name__ == "__main__":
    main()
