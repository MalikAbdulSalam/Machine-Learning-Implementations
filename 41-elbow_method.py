import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Load dataset
df = pd.read_csv("Wholesale customers data.csv")

# Select features
X = df.copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




inertias = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(
        model.inertia_
    )


plt.plot(
    range(2, 11),
    inertias,
    marker="*"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.show()

