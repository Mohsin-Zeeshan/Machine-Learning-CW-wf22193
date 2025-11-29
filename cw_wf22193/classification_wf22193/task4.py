# task4.py
from pathlib import Path
import pickle

import numpy as np
from sklearn.decomposition import PCA

RANDOM_SEED = 42
N_COMPONENTS = 200

DATA_DIR = Path("cifar-10-batches-py")
METHOD = "pca"          
OUTPUT_PATH = Path("cifar_pca_200.npz")


def load_batch(batch_path: Path):
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch[b"data"], np.array(batch[b"labels"])


def load_cifar10(data_dir: Path):
    X_train_parts, y_train_parts = [], []
    for i in range(1, 6):
        X, y = load_batch(data_dir / f"data_batch_{i}")
        X_train_parts.append(X)
        y_train_parts.append(y)

    X_train = np.concatenate(X_train_parts).astype(np.float32) / 255.0
    y_train = np.concatenate(y_train_parts)

    X_test, y_test = load_batch(data_dir / "test_batch")
    X_test = X_test.astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


def reduce_dimensions(X_train, X_test, method: str = "pca"):
    if method == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(X_train.shape[1], size=N_COMPONENTS, replace=False)
        return X_train[:, idx], X_test[:, idx]

    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    return X_train_reduced.astype(np.float32), X_test_reduced.astype(np.float32)


def load_reduced_data(reduced_path: Path):
    if not reduced_path.exists():
        raise FileNotFoundError(
            f"Could not find reduced dataset at {reduced_path}. Run task4.py first."
        )

    data = np.load(reduced_path)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_DIR}. Download and extract CIFAR-10 first."
        )

    X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)
    X_train_red, X_test_red = reduce_dimensions(X_train, X_test, METHOD)

    np.savez(
        OUTPUT_PATH,
        X_train=X_train_red,
        y_train=y_train,
        X_test=X_test_red,
        y_test=y_test,
    )
    print(
        f"Saved reduced data to {OUTPUT_PATH} "
        f"(train shape {X_train_red.shape}, test shape {X_test_red.shape})."
    )


if __name__ == "__main__":
    main()
