import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


def tune_decision_tree(X_train, y_train):
    tree = DecisionTreeClassifier(random_state=RANDOM_SEED)
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 10, 50],
        "min_samples_leaf": [1, 5, 10],
    }

    search = GridSearchCV(
        estimator=tree,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def main():
    parser = argparse.ArgumentParser(
        description="Decision tree classifier on reduced CIFAR-10."
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
    args = parser.parse_args()

    import task4

    X_train_full, y_train, X_test_full, y_test = task4.load_cifar10(args.data_dir)
    X_train, X_test = task4.reduce_dimensions(
        X_train_full, X_test_full, method=args.method
    )

    search = tune_decision_tree(X_train, y_train)
    best_tree = search.best_estimator_

    print("Best hyperparameters:", search.best_params_)
    print(f"Cross-validation accuracy: {search.best_score_:.4f}")

    y_test_pred = best_tree.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
