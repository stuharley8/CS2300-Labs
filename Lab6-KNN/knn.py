import random

import numpy as np
import pandas as pd
from scipy import spatial
from scipy import stats

class KNN:
    """
    Implementation of the k-nearest neighbors algorithm for classification.
    FIXME - This implementation only uses sepal length and width instead of all the data columns to create the model, as I misinterpreted the instructions.
    """
    def __init__(self, k):
        """
        Takes one parameter.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point. 
        """
        self.k = k
        
    def set_k(self, k):
        """
        Set method for k.
        """
        self.k = k
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        """
        self.iris = X
        self.target_name = y
        
    def predict_loop(self, X):
        """
        Predicts the output variable's values for the query points X using loops.
        
        """
        # X[0] is length
        # X[1] is width
        predicted = np.empty(len(X), dtype='<U10')
        for x in range(len(X)):
            distances = pd.DataFrame()
            distances["distance"] = spatial.distance.cdist([[X[x,0],X[x,1]]], self.iris[:,[0,1]])[0]
            distances["label"] = self.target_name
            sorted_distances = distances.sort_values("distance").iloc[:self.k]
            label = stats.mode(sorted_distances["label"].values).mode
            predicted[x] = label[0]
        try:
            return predicted.astype(int)
        except:
            return predicted
        
    def predict_numpy(self, X):
        """
        Predicts the output variable's values for the query points X using numpy (no loops).
        """
        distances = spatial.distance.cdist(X[:,0:2], self.iris[:,0:2]) # calculates the distances between every point in X and every point in the iris data. Returns a matrix shape (len(X) rows * len(iris.data) columns)
        closest = np.argsort(distances, axis=1).transpose()[:self.k].transpose() # creates a sorted matrix of the indexes of the smallest k distances. Returns a matrix shape (len(X) rows * k colummns)
        targets = np.take(self.target_name, closest) # creates a matrix of target_names associated with the indexes from the previous matrix. Returns a matrix shape (len(X) rows * k colummns)
        return stats.mode(targets, axis=1).mode.flatten() # creates an array of the modes which represents the predicted label for each query point. Returns a matrix shape (1 row * len(X) columns)