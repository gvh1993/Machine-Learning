# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:03:35 2017

@author: Gert-Jan
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

accuracies = []
for i in range(25):

    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    
    #replace questionmarks
    df.replace('?', -99999, inplace=True)
    
    #remove unusefull columns
    df.drop(['id'], 1, inplace=True)
    
    
    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])
    
    
    #create test and training data from X and y
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    
    
    #define classifier
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    accuracies.append(accuracy)
print(sum(accuracies)/len(accuracies))
#print(accuracy)
#
#example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,3,2,5,2,1]])
#example_measures = example_measures.reshape(len(example_measures), -1)
#prediction = clf.predict(example_measures)
#
#print(prediction)