import pandas as pd
from sklearn.cluster import KMeans
import joblib


data = {
    "Age": [20, 22, 25, 45, 48, 50, 23, 27],

    "Income": [
        20000,
        22000,
        25000,
        60000,
        65000,
        70000,
        24000,
        28000
    ]
}

df = pd.DataFrame(data)

model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(
    df[["Age", "Income"]]
)

# print(df)

# Save trained model
joblib.dump(model, "./customer_cluster.pkl")


# predict on new data

new_customer = [[240, 260000]]
cluster = model.predict(new_customer)
print(cluster)