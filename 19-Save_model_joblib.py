import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Example dataset
data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "salary": [20, 25, 30, 35, 40, 45, 50, 55]
}

df = pd.DataFrame(data)


# Features and target
X = df[["hours"]]
y = df["salary"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Evaluate model
print("R2:", r2_score(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))


# Save trained model
joblib.dump(model, "regression_model.pkl")

print("Regression model saved!")