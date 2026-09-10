
# importing pandas for data reading
import pandas as pd

# importing min_max scaler
from sklearn.preprocessing import MinMaxScaler

# Read CSV file
df = pd.read_csv("data.csv")
# print(df)

# Create scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the Age column
df["Math"] = scaler.fit_transform(df[["Math"]])
print(df["Math"])

# Scale the Age column
df["Age"] = scaler.fit_transform(df[["Age"]])
print(df["Age"])

