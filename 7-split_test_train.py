import pandas as pd
from sklearn.model_selection import train_test_split

# Read CSV file
df = pd.read_csv("data.csv")
# print(df)

# print(df.head())

print("____________________________________________________")

# X are features
# Y are labels
# Y hat predictions

X = df[["StudentID", "Name" , "Age" , "Gender" , "Math" ,  "Physics", "Chemistry", "English", "Attendance"]]
Y = df["Result"]

# print(X)
print("____________________________________________________")
# print(Y)




# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)
print(X_train)
print("____________________________________________________")
print(y_train)
print("____________________________________________________")
print(X_test)
print("____________________________________________________")
print(y_test)