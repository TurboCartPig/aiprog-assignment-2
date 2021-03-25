import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Set pyplot theme from seaborn
sns.set_theme()

# Make up some synthetic data
X, targets = make_blobs(n_samples=1000, centers=3)

# Create a kmeans model and fit it to the data
kmeans = KMeans(init="k-means++", n_clusters=3)
kmeans.fit(X, targets)

# Predict the cluster of each point in the data
P = kmeans.predict(X)

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], picker=True, c=P)
plt.title("Clusters")
plt.show()

# But what if we don't know the number of clusters?
# Use the elbow method

# Some settings
max_k = 10

# Make a synthetic dataset with 5 clusters
X, targets = make_blobs(n_samples=1200, centers=5)

models = [KMeans(n_clusters=i).fit(X) for i in range(1, max_k)]
inertias = [model.inertia_ for model in models]
scores = [model.score(X) for model in models]
print(inertias)
print(scores)

# Plot the inertias for ks to visualize the elbow curve
plt.plot(range(1, max_k), inertias)
plt.title("Elbow curve")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()