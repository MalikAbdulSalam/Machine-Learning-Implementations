from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as owais
import matplotlib.pyplot as plt


# Read CSV file
df = owais.read_csv("big_data.csv")
print(df)



X = df.drop("Price", axis=1)

y = df["Price"]


X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42

)


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)

print("R²:", r2)








residuals = y_test - y_pred

plt.scatter(y_pred, residuals)

plt.axhline(0)

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.show()