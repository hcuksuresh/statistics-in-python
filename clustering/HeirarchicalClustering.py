# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 13:22:18 2017

@author: BGH47373
"""

from scipy.cluster.hierarchy import dendrogram,linkage
from matplotlib import pyplot as plt

import pandas as pd

cars = pd.read_csv(r'C:\Users\bgh47373\Documents\MyPythonCodes_Study\DataSet\Ex_Files_Python_Data_Science_EssT\Ex_Files_Python_Data_Science_EssT\Exercise Files\Ch06\06_02\mtcars.csv')
cars.columns = ['car_names','mpg','cyl','disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']


# parameters that can determine automatic or not
X = cars.ix[:,(1,3,4,6)].values

# Automatic or not
y = cars.ix[:,(9)].values

Z = linkage(X,'ward')

dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

plt.axhline(y=500)
plt.axhline(y=150)
plt.show()


# Lets use Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
HCluster = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
HCluster.fit(X)
sm.accuracy_score(y,HCluster.labels_)
