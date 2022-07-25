################################################################################
# Author 1:      Martin Scheffauer
# MatNr 1:       51917931
# Author 2:      Dominik Geschwinde
# MatNr 2:       12108977
# File:          knn.py
# Description:  this file contains code for the knn classifier
# Comments:    the classifier searches the n_nearest neighbors and returns for each sample the predicted label of the neighbor.
#
#
################################################################################


import numpy as np
from collections import Counter as c


class kNN:
    def __init__(self, n_neighbors:int, metric:str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.cut = 0
    def __repr__(self) -> str:
        return f"kNN(n_neighbors={self.n_neighbors} metric={self.metric})"
    
    def fit(self, X_train:np.ndarray, y_train:np.ndarray):
        self._X_train = X_train
        self._y_train = y_train

    def _predict_sample(self, x) -> int:
        
        calculator = DistanceCalculator(self.metric)

        distances = [calculator.calculate(x, x_train_element) for x_train_element in self._X_train]
    
        indices = np.argsort(distances)[:self.n_neighbors]
      
        k_nearest_label = [self._y_train[i] for i in indices]
       
     
        #prepare a dictionary with the uniques and coutns       
        _uniques, counts = np.unique(k_nearest_label,return_counts = True)
       
        _dict = dict()
        i = 0
        for unique in _uniques:
            _dict[unique] = counts[i]
            i = i + 1
        #sort the dictionary    
        od = dict(sorted(_dict.items(), key=lambda item: item[1], reverse=True))
     
        _values = list()
        _old_count = list(od.values())[0]
  
        #take the element with the smallest class label if a draw occurs
        for value, count in od.items():
            if _old_count == count:
                _values.append(value)
            else:
                break    
     
        return min(_values)


    def predict(self, X_test:np.ndarray) -> np.ndarray:
        predicted_lables = [self._predict_sample(x) for x in X_test]
        return np.array(predicted_lables)


class DistanceCalculator:
    def __init__(self, metric:str) -> None:
        self.metric=metric
    
    def calculate(self,p,q) -> float:
        if self.metric == "cosine":
            return self._cosine(p,q)

        elif self.metric == "euclidean":
            return self._euclidean(p,q)

        elif self.metric == "manhattan":
            return self._manhattan(p,q)

        elif self.metric == "chebyshev":
            return self._chebyshev(p,q)
    
    def _cosine(self,p,q):
        return (1.0 - (np.dot(p,q)/(np.linalg.norm(p)*np.linalg.norm(q))))
        
    def _euclidean(self,p,q):
        return np.sqrt(np.sum(np.square(p-q)))

    def _manhattan(self,p,q):
        return sum(abs(value1 - value2) for value1, value2 in zip(p,q))
        
    def _chebyshev(self,p,q):
        return max(abs(value1 - value2) for value1, value2 in zip(p,q))

