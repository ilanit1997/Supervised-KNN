import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd
import collections
from sklearn.neighbors import KNeighborsClassifier


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p
        self.training = None
        self.train_labels = None


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        self.training = X
        self.train_labels = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        predicted_labels = np.array(list(map(self._predict, X)))

        return predicted_labels

    def _predict(self, x):
        """
        :param x: point to predict label
        :return: label prediction
        """
        distance = np.array(list(map(lambda y: sum(abs(y-x)**self.p)**(1./self.p),
                                     self.training)))
        indices = distance.argsort()
        distance = np.sort(distance)
        labels = self.train_labels[indices]
        dist_labels = list(map(lambda x, y: (x, y), distance, labels))
        k_ng = dist_labels[:self.k]

        #Breaking Ties
        #break ties when k and next nghs are same distance from x
        k_dist = k_ng[-1]  #take k ngh
        #take nghs from k forward which have same distance as k_dist
        k_1 = list(filter(lambda x: x not in k_ng and x[0] == k_dist[0], dist_labels))
        #add k_ngh to k_1 list and take ngh with smaller lexicographic order
        k_1.append(k_dist)
        k2 = sorted(k_1, key=lambda x:x[1])[0]
        k_ng.pop() #remove last element
        k_ng.append(k2) #add correct k neightboor


        #break ties when majority labels have same freq
        (unique, count) = np.unique(np.array(k_ng)[:,1], return_counts=True)
        freq = sorted(np.asarray((unique, count)).T, key=lambda x: x[1], reverse=True)
        max_freq = freq[0][1]
        max_labels = np.array(list(filter(lambda x: x[1]== max_freq, freq))).T[0]
        max_labels_points = np.array(list(filter(lambda x: x[1] in max_labels, k_ng)))
        min_dist = max_labels_points[0][0]
        min_dist_points = np.array(list(filter(lambda x: x[0] == min_dist, max_labels_points)))
        prediction = sorted(min_dist_points, key=lambda x:x[1])[0][1]
        return prediction

        


def main():

    print("*" * 20)
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
