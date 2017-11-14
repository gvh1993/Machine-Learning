# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 23:09:45 2017

@author: Gert-Jan
"""

import tensorflow as tf
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def part1():
    
    features = [[140, 1], [130, 1], [150, 0], [170, 0]] # 0 = bumpy, 1 = smooth
    labels = [0, 0, 1, 1] # 0 = apple, 1 = orange
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, labels)
    
    print(clf.predict([[160, 0]]))

def part2():
    iris = load_iris()
    #print(iris.feature_names)
    #print(iris.target_names)
    #print(iris.data[0])
    
    test_idx = [0,50,100]
    
    #training data
    train_target = np.delete(iris.target, test_idx)
    train_data = np.delete(iris.data, test_idx, axis=0)
    
    #testing data
    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)
    
    print(test_target)
    print(test_data)
    print(clf.predict(test_data))
    
    
def part3():
    greyhouds = 500
    labs = 500
    
    grey_height = 28 + 4 * np.random.randn(greyhouds)
    lab_height = 24 + 4 * np.random.randn(labs)

    plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
    plt.show()
    
part3()