
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Read CSV file
df = pd.read_csv("data2.csv")
print(df)


print("###################################")

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["City"] = le.fit_transform(df["City"])

print(df)

