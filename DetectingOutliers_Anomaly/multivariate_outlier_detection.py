# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 23:44:10 2017

@author: BGH47373
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn as sb

dataset = "C:\\Users\\bgh47373\\Documents\\MyPythonCodes_Study\\DataSet\\Ex_Files_Python_Data_Science_EssT\\Ex_Files_Python_Data_Science_EssT\\Exercise Files\\Ch05\\05_01\\iris.data.csv"

df = pd.read_csv(dataset,header=None,sep=',')

df.columns = ['Sepal Length','Sepal Width', 'Petal Length','Petal Width','Species']

x=df.ix[:,:4].values
y=df.ix[:,4].values


# This is platting two variables species and sepal length together
sb.boxplot(x='Species',y='Sepal Length',data=df)

# or
sb.pairplot(df,hue='Species')