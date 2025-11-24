import argparse
from pathlib import Path
import warnings

import numpy as np
from joblib.externals.loky.process_executor import BrokenProcessPool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

RANDOM_SEED = 42


def subsample_data(X, y, limit):
    """Return a random subset of (X, y) if limit is set and smaller than data."""
    if limit is None or limit <= 0 or limit >= len(X):
        return X, y, False

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=limit, replace=False)
    return X[idx], y[idx], True


def tune_random_forest(
    X_train,
    y_train,
    cv_folds,
    search_iterations,
    search_jobs,
    forest_jobs,
    allow_retry=True,
):
    forest = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=forest_jobs,
    )
    param_distributions = {
        "n_estimators": [120, 160, 200, 240, 280],
        "max_depth": [20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        estimator=forest,
        param_distributions=param_distributions,
        n_iter=search_iterations,
        cv=cv_folds,
        n_jobs=search_jobs,
        verbose=1,
        random_state=RANDOM_SEED,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A worker stopped while some jobs were given to the executor.",
                category=UserWarning,
            )
            search.fit(X_train, y_train)
    except BrokenProcessPool:
        if allow_retry and (search_jobs != 1 or forest_jobs != 1):
            print(
                "Parallel workers crashed during tuning; "
                "retrying with single-job execution to stay stable."
            )
            return tune_random_forest(
                X_train,
                y_train,
                cv_folds,
                search_iterations,
                search_jobs=1,
                forest_jobs=1,
                allow_retry=False,
            )
        raise
    return search, forest_jobs


def main():
    parser = argparse.ArgumentParser(
        description="Random forest ensemble on reduced CIFAR-10."
    )
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
        default=20000,
        help="Number of training examples to use for hyperparameter search (<=0 means all).",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=3,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--search_iterations",
        type=int,
        default=20,
        help="Number of sampled hyperparameter settings for RandomizedSearchCV.",
    )
    parser.add_argument(
        "--search_jobs",
        type=int,
        default=1,
        help="Parallel jobs for hyperparameter search (keep modest to avoid OOM).",
    )
    parser.add_argument(
        "--forest_jobs",
        type=int,
        default=1,
        help="Parallel jobs for the random forest estimator.",
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
            f"out of {len(X_train)} for stability."
        )

    search, effective_forest_jobs = tune_random_forest(
        X_tune,
        y_tune,
        cv_folds=args.cv_folds,
        search_iterations=args.search_iterations,
        search_jobs=args.search_jobs,
        forest_jobs=args.forest_jobs,
    )
    best_params = search.best_params_

    print("Best hyperparameters:", best_params)
    print(f"Cross-validation accuracy: {search.best_score_:.4f}")

    best_forest = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=effective_forest_jobs,
        **best_params,
    )
    best_forest.fit(X_train, y_train)

    y_test_pred = best_forest.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
