# task7.py
# READ ME:
# Decision Tree Classifier on PCA-reduced CIFAR-10 from task4.py
# Do task5.py after task7.py to ensure reduced data is available
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

REDUCED_DIR = Path("cifar_pca_200.npz")
MAX_TRAIN_SAMPLES = 15000  # Limit for grid search due to SVM training time


def load_reduced_data(path: Path):
    """Load PCA-reduced CIFAR-10 dataset"""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find reduced dataset at {path}. Run task4.py first."
        )
    data = np.load(path)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def main():
    X_train, y_train, X_test, y_test = load_reduced_data(REDUCED_DIR)

    if X_train.shape[0] > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_train.shape[0], size=MAX_TRAIN_SAMPLES, replace=False)
        X_train_cv = X_train[idx]
        y_train_cv = y_train[idx]
    else:
        X_train_cv = X_train
        y_train_cv = y_train

#Define hyperparameter grid for RBF kernel
    param_grid = [
        {
            "kernel": ["rbf"],
            "C": [2.5, 3, 3.5, 4],
            "gamma": [0.008, 0.01, 0.012, 0.015, "scale"],
        },
    ]

    svm = SVC(random_state=42, cache_size=1000)

#Perform grid search with cross-validation on subset
    clf = GridSearchCV(
        svm,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    clf.fit(X_train_cv, y_train_cv)

    print("Best params (SVM):", clf.best_params_)
    print(f"Best CV score: {clf.best_score_:.4f}")

#Train final model on full training set with best parameters
    best_params = clf.best_params_
    final_svm = SVC(
        C=best_params['C'],
        gamma=best_params['gamma'],
        kernel=best_params['kernel'],
        random_state=42,
        cache_size=1000
    )
    final_svm.fit(X_train, y_train)

#Evaluate on test set
    y_pred = final_svm.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (SVM): {test_acc:.4f}")


if __name__ == "__main__":
    main()
