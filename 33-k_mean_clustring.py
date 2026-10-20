from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([
    [1, 2],
    [2, 1],
    [2, 3],
    [8, 8],
    [9, 8],
    [8, 9]
])

x = (1,2,2,8,9,8)
y = (2,1,3,8,8,9)

plt.scatter(x,y)
plt.show()

# Create K-Means model
model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

# Train the model
model.fit(X)

# Get cluster labels
labels = model.labels_

print(labels)