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


# Train final model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# Save results
df.to_csv("customer_clusters.csv", index=False)


joblib.dump(kmeans, "./customer_project_trained_model.pkl")