# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:58:37 2017

@author: sarwesh suman

@description: random forest classifier example code
"""

import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from  sklearn.metrics import confusion_matrix 

file_name = r'C:\Users\bgh47373\Documents\Module -2\part1\classification\diabetes.csv'

df_diabities = pd.read_csv(file_name)

print(df_diabities.head())

""" 
This is sample data of diabitic patients, the goal it to predict if the patient is diabitic or not 
If the patient is diabetic then the outcome will be 1, there are two classes only, 0 or 1
"""

# Pre processing steps

# shuffling because we dont want data in serial order
df_diabities = shuffle(df_diabities)

X=df_diabities.ix[:,:-1] # Last column is Outcome or class to predict
Y=df_diabities.ix[:,-1] # Outcome with value 0 or 1

# Lets create train test data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

# Building the model

rf_classifier = RandomForestClassifier(n_estimators = 200 , criterion='entropy') # criterion is used to select the feature which best classifies the dataset

# Lets train
rf_classifier.fit(X_train,Y_train)

# Lets now predict
Y_pred=rf_classifier.predict(X_test)

# Lets compare

print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

# Lets get sensitivity or  specificity manually

cmatrix = confusion_matrix(Y_test,Y_pred)
print(cmatrix)

#generic matrix
tp, fp, fn, tn = cmatrix.ravel()

print("Sensitivity = ",tp/(tp+fn))
print("Specificity =", tn/(tn+fp))
print("Accuracy=",(tp+fn)/(tp+fn+fp+fn))
