# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:38:24 2017

@author: sarwesh suman

KNN regression for boston data

"""

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = load_boston()

features = data.data
target = data.target

x_train,x_test,y_train,y_test = train_test_split(features,target,train_size=0.7)

model = KNeighborsRegressor(n_neighbors=6)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("mean_squared_error = ",mean_squared_error(y_test,y_pred))
print("r2_score = " , r2_score(y_test,y_pred))