import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import graphviz



X_mnist, y_mnist = fetch_openml(name='mnist_784',return_X_y=True, as_frame=False)
frac_of_dataset = 0.5
index = int(frac_of_dataset*X_mnist.shape[0])
X_train, X_test, y_train, y_test = train_test_split(X_mnist[:index,:], y_mnist[:index], test_size=0.2, random_state=5)
