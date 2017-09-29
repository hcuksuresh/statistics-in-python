# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:02:29 2017

@author: sarwesh suman
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = load_boston()

features = data.data
target = data.target

x_train,x_test,y_train,y_test = train_test_split(features,target,train_size=0.7)

lr = LinearRegression()
ridge = Ridge(alpha=0.1)

lr.fit(x_train,y_train)
ridge.fit(x_train,y_train)

lr_pred = lr.predict(x_test)
ridge_pred = ridge.predict(x_test)

print("\nModel coeefficients.....\nlr,ridge")
for i,val in enumerate(lr.coef_):
    print("{}\t{}".format(val,ridge.coef_[i]))

print("\nW0 = {}\t{} ".format(lr.intercept_,ridge.intercept_))

print("\nmean_squared_error = {}\t{}".format(mean_squared_error(y_test,lr_pred),mean_squared_error(y_test,ridge_pred)))
print("\nr2_score = {}\t{}".format(r2_score(y_test,lr_pred),r2_score(y_test,ridge_pred)))

print(""" to individually extracting a column 
data.data[:5,(data.feature_names == 'TAX').astype('int').argmax()]
""")