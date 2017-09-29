# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:45:22 2017

@author: sarwesh suman

Lasso = L1 regularization

"""

import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error, r2_score

file_name=r'C:\Users\bgh47373\Documents\Module -2\part2\regression\smallerDetectionRegression.csv'

df = pd.read_csv(file_name)

print(df.head())

"""

  Senseout  operation3  sensor1  sensor2  sensor3  sensor4  sensor5  sensor6  \
0   9672.78       100.0   449.44   555.32  1358.61  1137.23     5.48     8.00   
1   9568.25       100.0   445.00   549.90  1353.22  1125.78     3.91     5.71   
2   9224.58        60.0   462.54   537.31  1256.76  1047.45     7.05     9.02   
3   9567.80       100.0   445.00   549.51  1354.03  1126.38     3.91     5.71   
4   9215.21        60.0   462.54   537.07  1257.71  1047.93     7.05     9.03   

   sensor7  sensor8  sensor9  sensor10  
0   194.64  2222.65  8341.91      1.02  
1   138.51  2211.57  8303.96      1.02  
2   175.71  1915.11  8001.42      0.94  
3   138.46  2211.58  8303.96      1.02  
4   175.05  1915.10  7993.23      0.94  

"""

features = df.ix[:,1:]
target = df['Senseout']

x_train,x_test,y_train,y_test = train_test_split(features,target,train_size=0.7)

"""
From wikipedia:
In statistics and machine learning, lasso (least absolute shrinkage and selection operator) 
(also Lasso or LASSO) is a regression analysis method that performs both variable selection 
and regularization in order to enhance the prediction accuracy and interpretability of the 
statistical model it produces.

"""
model = Lasso(alpha=0.5)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Model coeefficient ",model.coef_)
print("W0 = ",model.intercept_)

print("mean_squared_error = ",mean_squared_error(y_test,y_pred))
print("r2_score = " , r2_score(y_test,y_pred))

"""

Model coeefficient  
 [  1.47304457e+01   1.11226669e+01   4.55136036e+00  -1.04350665e+00
   3.85897306e-01   1.88946665e-01   1.54090781e+01  -2.62768423e-04
  -1.73288015e-01   1.62743976e-01  -8.44498281e-02]
W0 =  564.69628558
mean_squared_error =  114.560746249
r2_score =  0.999907325821

"""
