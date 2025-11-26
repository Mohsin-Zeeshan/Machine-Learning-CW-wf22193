import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_SEED = 42


def subsample_data(X, y, limit):
    if limit is None or limit <= 0 or limit >= len(X):
        return X, y, False

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=limit, replace=False)
    return X[idx], y[idx], True


def build_svm_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(random_state=RANDOM_SEED)),
        ]
    )


def svm_param_grid():
    return [
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.1, 1.0, 5.0],
        },
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.5, 1.0, 5.0],
            "svc__gamma": ["scale", 0.01, 0.001],
        },
        {
            "svc__kernel": ["poly"],
            "svc__C": [0.5, 1.0, 5.0],
            "svc__gamma": ["scale", 0.01],
            "svc__degree": [2, 3],
        },
    ]


def tune_svm(X_train, y_train, cv_folds, search_jobs):
    pipeline = build_svm_pipeline()
    param_grid = svm_param_grid()
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        n_jobs=search_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def main():
    parser = argparse.ArgumentParser(description="SVM classifier on reduced CIFAR-10.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("cifar-10-batches-py"),
        help="Directory containing CIFAR-10 python batches.",
    )
    parser.add_argument(
        "--method",
        choices=["pca", "random"],
        default="pca",
        help="Dimensionality reduction method (delegates to task4).",
    )
    parser.add_argument(
        "--tune_limit",
        type=int,
        default=10000,
        help="Training examples to use during hyperparameter search (<=0 means all).",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=3,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--search_jobs",
        type=int,
        default=1,
        help="Parallel jobs for GridSearchCV (SVMs are memory intensive).",
    )
    args = parser.parse_args()

    import task4

    X_train_full, y_train, X_test_full, y_test = task4.load_cifar10(args.data_dir)
    X_train, X_test = task4.reduce_dimensions(
        X_train_full, X_test_full, method=args.method
    )

    X_tune, y_tune, was_subsampled = subsample_data(X_train, y_train, args.tune_limit)
    if was_subsampled:
        print(
            f"Tuning hyperparameters on a subset of {len(X_tune)} samples "
            f"out of {len(X_train)} to keep SVM training practical."
        )

    search = tune_svm(
        X_tune,
        y_tune,
        cv_folds=args.cv_folds,
        search_jobs=args.search_jobs,
    )
    best_params = search.best_params_
    print("Best hyperparameters:", best_params)
    print(f"Cross-validation accuracy: {search.best_score_:.4f}")

    final_model = build_svm_pipeline()
    final_model.set_params(**best_params)
    final_model.fit(X_train, y_train)

    y_test_pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
