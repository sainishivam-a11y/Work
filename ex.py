from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv("Train.csv")

y = pd.read_csv("train_one.csv")

x_test = pd.read_csv("Test.csv")
labels = np.array(y)
print(labels)
[[ -82.15466656]
 [ -48.89796018]
 [  77.2703707 ]

 [-107.510508  ]
 [ -47.34155781]
 [-115.939003  ]]
from sklearn.neighbors import KNeighborsRegressor
>>> neigh = KNeighborsRegressor(n_neighbors=5)
>>> neigh.fit(X, y)