# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:05:48 2020

@author: james
"""
import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_excel(r"C:\Users\james\Documents\School\Machine Learning & Data mining\Project1\Data\hprice2.xls",header=None)

raw_data = df.to_numpy() 

'''PCA is done with X= all data, except 3 logarithm , even y =price '''
cols = range(0,9) 
X = raw_data[:,cols]
N,M= X.shape
attributeNames=['price','crime','nox','rooms','dist','radial', 'proptax', 'stratio',  'lowstat']

# Subtract the mean from the data
Y1 = X - np.ones((N, 1))*X.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2,0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

pca = PCA(n_components=2)
principle_comp = pca.fit_transform(Y2)

#principal = pd.DataFrame(data = principle_comp
#             , columns = ['principal component 1', 'principal component 2'])

principal = pd.DataFrame(data = principle_comp)
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('PC1',fontsize=20)
plt.ylabel('PC2',fontsize=20)
plt.title("Principal Component Analysis of HPRICE2 dataset",fontsize=20)
plt.scatter(principle_comp[:,0], principle_comp[:,1])


pca = PCA(n_components=5)
principle_comp = pca.fit_transform(Y2)

#principal = pd.DataFrame(data = principle_comp
#             , columns = ['principal component 1', 'principal component 2'])

principal = pd.DataFrame(data = principle_comp)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('PC1',fontsize=10)
ax.set_ylabel('PC2',fontsize=10)
ax.set_zlabel('PC3',fontsize=10)
#ax.title("Principal Component Analysis of HPRICE2 dataset",fontsize=20)
ax.scatter(principle_comp[:,0], principle_comp[:,1], principle_comp[:,2])
#plt.savefig('scatter3d.png')

#print(pca.components_)
#
#pca = PCA(n_components=2)
#pca.fit(Y2)
#X_pca = pca.transform(Y2)
#print("original shape:   ", X.shape)
#print("transformed shape:", X_pca.shape)
#print(pca.components_)
#
#X_new = pca.inverse_transform(X_pca)
#plt.scatter(Y2[:, 0], Y2[:, 1], alpha=0.2)
#plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
#plt.axis('equal');
#def draw_vector(v0, v1, ax=None):
#    ax = ax or plt.gca()
#    arrowprops=dict(arrowstyle='->',
#                    linewidth=2,
#                    shrinkA=0, shrinkB=0)
#    ax.annotate('', v1, v0, arrowprops=arrowprops)
#
## plot data
#plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
#for length, vector in zip(pca.explained_variance_, pca.components_):
#    v = vector * 3 * np.sqrt(length)
#    draw_vector(pca.mean_, pca.mean_ + v)
#plt.axis('equal');