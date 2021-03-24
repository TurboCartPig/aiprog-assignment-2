"""Task 2.1 - K-means clustering given K."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import choice, random_sample


# Load iris dataset
def load_iris():
    return pd.read_csv(
        "./datasets/iris/iris.data",
        header=0,
        names=["sepal_lenght", "sepal_width", "petal_length", "petal_width", "class"],
    )


# Euclidean distance between two points.
def euclidean(a, b):
    return np.linalg.norm(a - b)


# Find the index of the centroid the point is closest to.
def shortest_distance(point, centroids):
    dists = np.array([euclidean(point, centroid) for centroid in centroids])
    return np.argmin(dists)


def kmeans(data, k, epochs):
    num_points, num_elems = data.shape

    # Pick k random points as initial centroids
    centroids_idx = choice(num_points, k, replace=False)
    centroids = data[centroids_idx, :]

    point_to_centroid = []

    for _ in range(epochs):
        # Find the centroid with shortest distance per point
        point_to_centroid = [
            shortest_distance(data[i], centroids) for i in range(num_points)
        ]

        for i in range(k):
            points_in_centroid = np.array(
                [data[j] for j in point_to_centroid if j == i]
            )
            centroids[i] = np.mean(points_in_centroid)

    return centroids, point_to_centroid


# K-means clustering given k clusters where x is points to cluster.
# def kmeans(data, k, epochs):
#     idx = choice(len(data), k, replace=False)

#     centroids = data[idx, :]

#     distances = [euclidean(centroids, i) for i in data]

#     points = np.array([np.argmin(i) for i in distances])

#     for _ in range(epochs):
#         centroids = []
#         for idx in range(k):
#             loccent = data[points == idx].mean(axis=0)
#             centroids.append(loccent)

#         centroids = np.vstack(centroids)
#         distances = [euclidean(centroids, i) for i in data]
#         points = np.array([np.argmin(i) for i in distances])

#     return points


# Load the dataset and split it into data and labels
ds = load_iris()
data = ds[["sepal_lenght", "sepal_width", "petal_length", "petal_width"]].to_numpy()
labels = ds[["class"]].to_numpy()

# Setup seaborn
sns.set_theme()

# Create and show a scatterplot of the dataset
sns.scatterplot(data=ds, x="sepal_lenght", y="sepal_width", hue="class")
plt.show()

clusters, labels = kmeans(data, 3, 10)
print(clusters)
print(labels)
# df = pd.DataFrame(clusters)
# print(df)
# sns.scatterplot(data=df)
# plt.show()

# points = kmeans(data, 3, 50)
# print(points)
