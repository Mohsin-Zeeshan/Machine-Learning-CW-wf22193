import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


def build_model(input_dim: int) -> nn.Module:
    # Small MLP with two hidden layers
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def prepare_data(file_name: str = "regression_insurance.csv"):
    data = pd.read_csv(file_name)
    X = data.drop(columns=["charges"])
    y = data["charges"].values.astype(np.float32)

    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Handle sparse output from OneHotEncoder
    def to_numpy(array_like):
        return array_like.toarray() if hasattr(array_like, "toarray") else np.asarray(array_like)

    return (
        torch.tensor(to_numpy(X_train_prep), dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(to_numpy(X_test_prep), dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )


def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 400, lr: float = 0.001, batch_size: int = 64):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    return model


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.sqrt(nn.functional.mse_loss(y_pred, y_true)).item()


def main():
    X_train, y_train, X_test, y_test = prepare_data()
    input_dim = X_train.shape[1]

    model = build_model(input_dim)
    model = train_model(model, X_train, y_train)

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        test_pred = model(X_test)

    train_rmse = rmse(y_train, train_pred)
    test_rmse = rmse(y_test, test_pred)

    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")

    # Scatter plot predicted vs actual on the test set
    y_test_np = y_test.squeeze().numpy()
    test_pred_np = test_pred.squeeze().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_np, test_pred_np, alpha=0.7, edgecolor="k")
    min_val = float(min(y_test_np.min(), test_pred_np.min()))
    max_val = float(max(y_test_np.max(), test_pred_np.max()))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal fit")
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title("Neural Network: Predicted vs Actual Charges (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()