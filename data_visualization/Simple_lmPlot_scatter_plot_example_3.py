# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:08:11 2017

@author: sarwesh suman
"""

import pandas as pd
import seaborn as sns

file_name = r'C:\Users\bgh47373\Documents\Module -2\part2\regression\mpg.csv'

df= pd.read_csv(file_name,header=0)

print(df.head())

"""

    mpg  cylinders  displacement horsepower  weight  acceleration  model_year  \
0  18.0          8         307.0        130    3504          12.0          70   
1  15.0          8         350.0        165    3693          11.5          70   
2  18.0          8         318.0        150    3436          11.0          70   
3  16.0          8         304.0        150    3433          12.0          70   
4  17.0          8         302.0        140    3449          10.5          70   

   origin                       name  
0       1  chevrolet chevelle malibu  
1       1          buick skylark 320  
2       1         plymouth satellite  
3       1              amc rebel sst  
4       1                ford torino  

"""

""" plotting mpg vs weight """

# Negative slope because as weight increases mile per galon decreases
sns.lmplot(x='weight',y='mpg',data=df)

""" plotting mpg vs cylinder """

# Negative slope because as cylinder incraeses mile per galon decreases 
sns.lmplot(x='cylinders',y='mpg',data=df)
