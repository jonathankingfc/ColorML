from math import sqrt
import random
import numpy as np


class Cluster:

    def __init__(self, center, points):
        self.center = center
        self.points = points


class KMeans:

    def __init__(self, n_clusters, min_diff=1):

        self.n_clusters = n_clusters
        self.min_diff = min_diff

        self.clusters = []

    def get_centers(self):
        return [cluster.center for cluster in self.clusters]

    def calculate_center(self, points):
        n_dim = len(points[0])
        vals = [0.0 for i in range(n_dim)]
        for p in points:
            for i in range(n_dim):
                vals[i] += p[i]
        coords = [(v / len(points)) for v in vals]
        return coords

    def assign_points(self, clusters, points):

        plists = [[] for i in range(self.n_clusters)]

        for p in points:
            smallest_distance = float('inf')

            for i in range(self.n_clusters):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i

            plists[idx].append(p)

        return plists

    def fit(self, points):
        random.seed(0)

        initial_centers = points[np.random.choice(
            points.shape[0], size=self.n_clusters), :]

        clusters = [Cluster(center=p, points=[p])
                    for p in initial_centers]

        while True:

            plists = self.assign_points(clusters, points)

            diff = 0

            for i in range(self.n_clusters):
                if not plists[i]:
                    continue
                old = clusters[i]
                center = self.calculate_center(plists[i])
                new = Cluster(center, plists[i])
                clusters[i] = new
                diff = max(diff, euclidean(old.center, new.center))

            if diff < self.min_diff:
                break

        clusters.sort(
            key=lambda c: len(c.points), reverse=True)

        self.clusters = clusters

    def transform(self, points):

        transformed_points = []

        for p in points:
            smallest_distance = float('inf')

            for i in range(self.n_clusters):
                distance = euclidean(p, self.clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i

            transformed_points.append(self.clusters[idx].center)

        return transformed_points


def euclidean(p, q):
    n_dim = len(p)
    return sqrt(sum([
        (p[i] - q[i]) ** 2 for i in range(n_dim)
    ]))
