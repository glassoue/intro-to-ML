#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:02:16 2020

@author: ghassen97
"""
import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from scipy.linalg import svd



#%%


df = pd.read_excel(r"/home/ghassen97/Desktop/S8/Intro to ML/project/dataset/hprice2.xls",header=None)

X_stat= df.iloc[:,0:9]
#X_stat.describe()
raw_data = df.to_numpy() 

'''PCA is done with X= all data, except 3 logarithm , even y =price '''
cols = range(0,9) 
X = raw_data[:,cols]
N,M= X.shape
attributeNames=['price','crime','nox','rooms','dist','radial', 'proptax', 'stratio',  'lowstat']
'standardize'
Xhat= (X -  np.ones((N,1))*X.mean(axis=0) ) # / ( X.std(axis=0) )
Xhat = Xhat*(1/np.std(Xhat,0))

# PCA by computing SVD of Y
U,S,Vh = svd(Xhat,full_matrices=False)
V=Vh.T
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 
Z = U*S;

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

print('Ran Project 1')
#%%
# We saw  that the first 5 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2,3,4]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b','y','p']
bw = .1
r = np.arange(1,M+1)
plt.figure(figsize=(8,8))
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()

















