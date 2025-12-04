# task5.py
# READ ME:
# Decision Tree Classifier on PCA-reduced CIFAR-10 from task4.py
# Do task5.py after task4.py to ensure reduced data is available
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

REDUCED_DIR = Path("cifar_pca_200.npz")


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

#Define hyperparameter search space
    params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
    }

#Perform grid search with cross-validation
    clf = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=params,
        cv=3,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("Best params:", clf.best_params_)

#Evaluate best model on test set
    best_tree = clf.best_estimator_
    y_pred = best_tree.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (decision tree): {test_acc:.4f}")


if __name__ == "__main__":
    main()
