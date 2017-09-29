# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:52:54 2017

@author: sarwesh suman
"""

import pandas as pd
import seaborn as sns

file_name=r'C:\Users\bgh47373\Documents\Module -2\part1\unsupervised\clustering\detection2.csv'

df = pd.read_csv(file_name)

print(df.head())

"""

unit number  time cycle  operation1  operation2  operation3  sensor1  \
0            1           1     34.9983      0.8400       100.0   449.44   
1            1           2     41.9982      0.8408       100.0   445.00   
2            1           3     24.9988      0.6218        60.0   462.54   
3            1           4     42.0077      0.8416       100.0   445.00   
4            1           5     25.0005      0.6203        60.0   462.54   

   sensor2  sensor3  sensor4  sensor5    ...     sensor13  sensor14  sensor15  \
0   555.32  1358.61  1137.23     5.48    ...      2387.72   8048.56    9.3461   
1   549.90  1353.22  1125.78     3.91    ...      2387.66   8072.30    9.3774   
2   537.31  1256.76  1047.45     7.05    ...      2028.03   7864.87   10.8941   
3   549.51  1354.03  1126.38     3.91    ...      2387.61   8068.66    9.3528   
4   537.07  1257.71  1047.93     7.05    ...      2028.00   7861.23   10.8963   

   sensor16  sensor17  sensor18  sensor19  sensor20  sensor21  sensor22  
0      0.02     334.0      2223    100.00     14.73    8.8071       NaN  
1      0.02     330.0      2212    100.00     10.41    6.2665       NaN  
2      0.02     309.0      1915     84.93     14.08    8.6723       NaN  
3      0.02     329.0      2212    100.00     10.59    6.4701       NaN  
4      0.02     309.0      1915     84.93     14.13    8.5286       NaN  

"""

""" plotting sensor3 vs sensor4 """

sns.lmplot(x='sensor3',y='sensor4',data=df)