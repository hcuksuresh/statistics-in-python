# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 17:48:38 2017

@author: BGH47373
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
import sklearn.metrics as sm

cars = pd.read_csv(r'C:\Users\bgh47373\Documents\MyPythonCodes_Study\DataSet\Ex_Files_Python_Data_Science_EssT\Ex_Files_Python_Data_Science_EssT\Exercise Files\Ch06\06_02\mtcars.csv')
cars.columns = ['car_names','mpg','cyl','disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']


# parameters that can determine automatic or not
X = cars.ix[:,(1,3,4,6)].values

# Automatic or not
y = cars.ix[:,(9)].values

x_scaled = scale(X)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=.33,random_state=17)

clf = KNeighborsClassifier()

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print(sm.classification_report(y_test,y_pred))