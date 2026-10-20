from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 2],
    [2, 1],
    [2, 3],
    [8, 8],
    [9, 8],
    [8, 9]
])

model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

model.fit(X)

labels = model.labels_

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

plt.xlabel("X")
plt.ylabel("Y")

plt.title("K-Means Clustering")

plt.show()