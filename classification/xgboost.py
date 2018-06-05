from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from numpy import genfromtxt
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
import sys
from sklearn.decomposition import SparsePCA,PCA

my_data = genfromtxt('training2.csv', delimiter=',',invalid_raise=False,dtype=None)

description_vectorizer = CountVectorizer(binary=True,decode_error='ignore')
description = description_vectorizer.fit_transform(my_data[:,0]).todense().tolist()

""" 
	Will Perform PCA on Fields which are having higher dimensions
"""

pca = PCA(n_components = 1320 )
description_pcaed = pca.fit_transform(description)

ce_reportedby_int = LabelEncoder().fit_transform(my_data[:,1]).reshape(-1,1)
ce_reportedby = OneHotEncoder(handle_unknown='ignore')
reportedby_onehot = ce_reportedby.fit_transform(ce_reportedby_int).todense().tolist()

pca_reportedby = PCA ( n_components = 100 )
reportedby_onehot_pcaed = pca_reportedby.fit_transform(reportedby_onehot)

ce_affectedperson_int = LabelEncoder().fit_transform(my_data[:,2]).reshape(-1,1)
ce_affectedperson = OneHotEncoder(handle_unknown='ignore')
ce_affectedperson_onehot = ce_affectedperson.fit_transform(ce_affectedperson_int).todense().tolist()

pca_affectedperson = PCA (n_components = 80 )
ce_affectedperson_pcaed = pca_affectedperson.fit_transform(ce_affectedperson_onehot)

ce_siteid_int = LabelEncoder().fit_transform(my_data[:,4]).reshape(-1,1)
ce_siteid = OneHotEncoder(handle_unknown='ignore')
ce_siteid_onehot = ce_siteid.fit_transform(ce_siteid_int).todense().tolist()

ce_assetid_int = LabelEncoder().fit_transform(my_data[:,5]).reshape(-1,1)
ce_assetid = OneHotEncoder(handle_unknown='ignore')
ce_assetid_onehot = ce_assetid.fit_transform(ce_assetid_int).todense().tolist()

ce_createdby_int = LabelEncoder().fit_transform(my_data[:,6]).reshape(-1,1)
ce_createdby = OneHotEncoder(handle_unknown='ignore')
ce_createdby_onehot = ce_createdby.fit_transform(ce_createdby_int).todense().tolist()

pca_createdby = PCA(n_components = 100 )
ce_createdby_pcaed = pca_createdby.fit_transform(ce_createdby_onehot)

X_arr = []
for idx,dsc in enumerate(description_pcaed):
	arr = []
	arr.extend(dsc)
	arr.extend(reportedby_onehot_pcaed[idx])
	arr.extend(ce_affectedperson_pcaed[idx])
	arr.extend(ce_siteid_onehot[idx])
	arr.extend(ce_assetid_onehot[idx])
	arr.extend(ce_createdby_pcaed[idx])
	X_arr.append(arr)


ce_ownergroup_int = LabelEncoder().fit_transform(my_data[:,3])

from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X_arr,ce_ownergroup_int,test_size=0.3)

import xgboost as xgb

xg_train = xgb.DMatrix(X_train, label=Y_train)

xg_test = xgb.DMatrix(X_test, label=Y_test)

param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.05
param['max_depth'] = 64
param['silent'] = 1
param['nthread'] = 16
param['num_class'] = 120

watchlist = [(xg_train, 'train'), (xg_test, 'test')]

num_round = 40

bst = xgb.train(param, xg_train, num_round, watchlist)

Y_pred = bst.predict(xg_test)

print(accuracy_score(Y_test,Y_pred))
