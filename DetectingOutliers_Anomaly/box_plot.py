# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 23:04:05 2017

@author: BGH47373
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot
from pylab import rcParams

dataset = "C:\\Users\\bgh47373\\Documents\\MyPythonCodes_Study\\DataSet\\Ex_Files_Python_Data_Science_EssT\\Ex_Files_Python_Data_Science_EssT\\Exercise Files\\Ch05\\05_01\\iris.data.csv"

df = pd.read_csv(dataset,header=None,sep=',')

df.columns = ['Sepal Length','Sepal Width', 'Petal Length','Petal Width','Species']

x=df.ix[:,:4].values
y=df.ix[:,4].values

df.boxplot()#return_type='dict')
pyplot.plot()

sepal_width_outliers = df[ (df.ix[:,1]>4 ) | (df.ix[:,1] < 2.2 )]