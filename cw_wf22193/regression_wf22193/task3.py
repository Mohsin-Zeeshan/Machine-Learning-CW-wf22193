#%%
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42


def prepare_data(file_name: str = "regression_insurance.csv"):
    data = pd.read_csv(file_name)
    X = data.drop(columns=["charges"])
    y = data["charges"].values.astype("float64")

    numeric = ["age", "bmi", "children"]
    categorical = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="first"), categorical),
        ]
    )

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    X_train_p = preprocessor.fit_transform(X_train)

    def to_dense(matrix):
        return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

    X_train_arr = to_dense(X_train_p).astype("float64")

    # Get and CLEAN feature names so they match task1.py style
    raw_feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = []
    for name in raw_feature_names:
        if "__" in name:
            name = name.split("__", 1)[1]  # drop "num__" / "cat__"
        clean_feature_names.append(name)

    return X_train_arr, y_train, clean_feature_names


def run_bayesian_regression(X_train: np.ndarray, y_train: np.ndarray):
    with pm.Model() as model:
        # Your original wide priors
        beta = pm.Normal("beta", mu=0.0, sigma=5000.0, shape=X_train.shape[1])
        intercept = pm.Normal("intercept", mu=10000.0, sigma=5000.0)
        sigma = pm.HalfNormal("sigma", sigma=5000.0)

        mu = intercept + pm.math.dot(X_train, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        trace = pm.sample(
            2000,
            tune=1000,
            chains=2,
            target_accept=0.9,
            random_seed=RANDOM_SEED,
        )

    return trace


def print_posterior_means(trace, feature_names):
    summary = pm.summary(
        trace,
        var_names=["intercept", "beta", "sigma"],
        round_to=3,
    )[["mean"]]

    print("Posterior means:")
    # Print coefficients feature-by-feature (same style as task1)
    for i, name in enumerate(feature_names):
        coef = summary.loc[f"beta[{i}]", "mean"]
        print(f"{name}: {coef:.3f}")

    # Intercept and noise at the end
    print(f"Intercept: {summary.loc['intercept', 'mean']:.3f}")
    print(f"sigma: {summary.loc['sigma', 'mean']:.3f}")


def main():
    X_train, y_train, feature_names = prepare_data()
    trace = run_bayesian_regression(X_train, y_train)
    print_posterior_means(trace, feature_names)


if __name__ == "__main__":
    main()
#%%
