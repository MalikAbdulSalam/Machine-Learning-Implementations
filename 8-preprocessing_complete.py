import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("data.csv")


# handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Physics"] =df["Physics"].fillna(df["Physics"].mean())



# convert strng into numbers
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Physics"] = df["Physics"].astype(int)
df["Age"] = df["Age"].astype(int)





# scale values
scaler = MinMaxScaler(feature_range=(-1, 1))
df[["Age", "Physics" , "Gender"]] = scaler.fit_transform(df[["Age", "Physics" , "Gender"]])
print(df)



# split data
X = df[["Age", "Physics" , "Gender"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:")
print(X_train)

print("Testing Data:")
print(X_test)




df.to_csv("preprcessed_data.csv", index=False)