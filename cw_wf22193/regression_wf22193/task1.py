#%%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Load the insurance dataset
file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)

X = data.drop(columns=["charges"])
y = data["charges"]

numerical_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(), categorical_features),
    ]
)

#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

model = LinearRegression()
model.fit(X_train_prep, y_train)

raw_feature_names = preprocessor.get_feature_names_out()
clean_feature_names = []
for name in raw_feature_names:
    if "__" in name:
        name = name.split("__", 1)[1]
    clean_feature_names.append(name)

#Display learned model parameters
print("Learned coefficients:")
for name, coef in zip(clean_feature_names, model.coef_):
    print(f"{name}: {coef:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

#Make predictions on both sets
y_train_pred = model.predict(X_train_prep)
y_test_pred = model.predict(X_test_prep)

#Calculate performance metrics
train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5

print(f"Train RMSE: {train_rmse:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")

#Create scatter plot of predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7, edgecolor="k")
#Draw diagonal line representing perfect predictions
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal fit")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Predicted vs Actual Insurance Charges (Test Set)")
plt.legend()
plt.tight_layout()
#Save visualisation to file
plt.savefig("task1_pred_vs_actual.png", dpi=300)
plt.show()
#%%
