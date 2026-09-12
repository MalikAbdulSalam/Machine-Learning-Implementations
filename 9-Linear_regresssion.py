import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# read data file
df = pd.read_csv("student_marks.csv")


# print data properties
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# Data split into 2 part
X = df[["Hours"]]   # Feature (independent)
y = df["Marks"]     # Target (dependent)



# data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% for testing
    random_state=42     # reproducible split
)



# create model
model = LinearRegression()


# train model
model.fit(X_train, y_train)


# predictions
y_pred_test = model.predict(X_test)

print("Predictions on test set:", y_pred_test)
print(X_test,y_test)


# Predict marks for a student who studied 12 hours
new_hours = [[10]]
predicted_marks = model.predict(new_hours)

print("Predicted marks for 11 hours of study:" , predicted_marks[0])


