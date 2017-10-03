# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:29:39 2017

@author: sarwesh suman
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from  sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import os

file_name = r'C:\Users\bgh47373\Documents\Module -2\part1\classification\diabetes.csv'

df_diabities = pd.read_csv(file_name)

print(df_diabities.head())

# Pre processing steps
# shuffling because we dont want data in serial order
df_diabities = shuffle(df_diabities)

X=df_diabities.ix[:,:-1] # Last column is Outcome or class to predict
Y=df_diabities.ix[:,-1] # Outcome with value 0 or 1

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7)

""" Rememeber that if the max depth is high then if the dataset is able to be
classified properly before 30 iterations of the model or estimator then
not all 30 estimators are created , for example if max_depth = 30 and
we are able to get good classification after 1st interation only then only 
1 estimator will be created. """
dt = DecisionTreeClassifier(max_depth = 1)
model = AdaBoostClassifier(n_estimators=30,base_estimator=dt)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("accuracy score ",accuracy_score(y_test,y_pred))

cmatrix = confusion_matrix(y_test,y_pred)
print(cmatrix)

tp, fp, fn, tn = cmatrix.ravel()

print("Sensitivity = ",tp/(tp+fn))
print("Specificity =", tn/(tn+fp))

print("ErrorS for each 30 DT trees ",model.estimator_errors_ )
print("Weights assigned to each estimators ",model.estimator_weights_)

""" Lets generate DT plot of each 30 runs """

""" Remember there might be less than 30 estimators """

from sklearn import tree
for i,est in enumerate(model.estimators_):
    tree.export_graphviz(est, out_file='new_tree.dot')
    os.system("dot -Tpng new_tree.dot -o output-images/tree"+str(i)+".png")