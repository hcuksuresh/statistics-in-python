# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 23:58:19 2017

@author: BGH47373
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot

import seaborn as sb

from sklearn.cluster import DBSCAN

from collections import Counter

dataset = "C:\\Users\\bgh47373\\Documents\\MyPythonCodes_Study\\DataSet\\Ex_Files_Python_Data_Science_EssT\\Ex_Files_Python_Data_Science_EssT\\Exercise Files\\Ch05\\05_01\\iris.data.csv"

df = pd.read_csv(dataset,header=None,sep=',')

df.columns = ['Sepal Length','Sepal Width', 'Petal Length','Petal Width','Species']

x=df.ix[:,:4].values
y=df.ix[:,4].values


model = DBSCAN(eps=0.8,min_samples = 19).fit(x)

print(model)

outlier_df = pd.DataFrame(x)
print(outlier_df[model.labels_ == -1])

# Lets see some visualization

fig = pyplot.figure()
ax = fig.add_axes([.1,.1,1,1])
colors = model.labels_

ax.scatter(x[:,2],x[:,1],c=colors,s=120) # s = size of the point
ax.set_xlabel('Petal Lendth')
ax.set_ylabel('Sepal Width')
pyplot.title('DB Scan collective outlier detection')
    

