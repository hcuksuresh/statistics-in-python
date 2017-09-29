# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:21:42 2017

@author: sarwesh suman

"""
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
import os

data = load_boston()

features = data.data
target = data.target

x_train,x_test,y_train,y_test = train_test_split(features,target,train_size=0.7)

model = DecisionTreeRegressor(min_samples_leaf=5,random_state=15) # Using max_depth and min_samples_leaf to control the sie of the model else it will explode with the dataset we have
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print('parameters: ', model.get_params)

print("mean_squared_error = ",mean_squared_error(y_test,y_pred))
print("r2_score = " , r2_score(y_test,y_pred))


#Outputting Decision Tree
tree.export_graphviz(model,out_file='treemtc.dot')

#Need conda install graphviz
os.system("dot -Tpng treemtc.dot -o tree.png")
