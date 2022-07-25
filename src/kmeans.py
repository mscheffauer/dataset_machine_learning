################################################################################
# Author 1:      Martin Scheffauer
# MatNr 1:       51917931
# Author 2:      Dominik Geschwinde
# MatNr 2:       12108977
# File:          kmeans.py
# Description:  this file contains code for the kmeans classifier
# Comments:    centroids get randomly initiated, distance from samples gets calculated
#
#
################################################################################


import numpy as np

class KMeans:
    def _euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def __init__(self,n_clusters:int = 8, max_iter:int = 300, tol:float = 0.0001, random_state:int = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
       

    def _init_randomizer(self):
        if self.random_state is None or self.random_state is np.random:
            return np.random.mtrand._rand
        elif isinstance(self.random_state,int):
            return np.random.RandomState(self.random_state)
        else:
            raise ValueError("random_state cannot be used to generate random number.")

    def __repr__(self) -> str:
        return f"KMeans(n_clusters={self.n_clusters}, max_iter={self.max_iter}, "\
        f"tol={self.tol}, random_state={self.random_state})"

    def fit(self,X_train:np.ndarray):
        centers = self._initialize_centroids(X_train,self.n_clusters)

        #init randomly the centers
        #random_obj = self._init_randomizer()
       
        #centers = random_obj.choice(X_train,size=self.n_clusters)
        for n in range(self.max_iter):
            closest = self._closest_centroid(X_train,centers)
            old_centers = centers
            centers = self._move_centroids(X_train,closest,centers)
            distances = [self._euclidean_distance(old_centers[i], centers[i]) for i in range(self.n_clusters)]
       
            if (sum(distances) < self.tol):
                break
        self._centers = centers

    def predict(self, X_test:np.ndarray) -> np.ndarray:
        return np.array([self._closest_centroid_of_sample(sample,self._centers) for sample in X_test])
   
    def _initialize_centroids(self,points, k):
        #init randomly the centers
        random_obj = self._init_randomizer()
       
        centroids = points.copy()
        random_obj.shuffle(centroids)
        return centroids[:k]

    def _closest_centroid(self,points, centroids):
    
        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _move_centroids(self,points, closest, centroids):
   
        return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

    def _closest_centroid_of_sample(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [self._euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index
