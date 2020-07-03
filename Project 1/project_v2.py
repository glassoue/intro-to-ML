#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:56:43 2020

@author: ghassen97
"""

import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from scipy.linalg import svd



#%%
#doc = xlrd.open_workbook('/home/ghassen97/Desktop/S8/Intro to ML/project/dataset/hprice2.xls').sheet_by_index(0)

# Extract attribute names (1st row, column 4 to 12)
#attributeNames = doc.row_values(0, 3, 11)

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

# Store the two in a cell, so we can just loop over them:
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot

plt.subplots_adjust(hspace=.4)
plt.title('NanoNose: Effect of standardization')
nrows=3
ncols=2

U1, S1, V1 = svd(Ys[0],full_matrices=False)
U2, S2, V2 = svd(Ys[1],full_matrices=False)

V1 = V1.T
V2 = V2.T

for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum() 
    
    # Compute the projection onto the principal components
    Z = U*S;
        
    # Plot attribute coefficients in principal component space
    plt.figure(figsize=(25,25))
    
    plt.subplot(nrows, ncols,  3+k)
    for att in range(V.shape[1]):
        plt.arrow(0,0, V[att,i], V[att,j])
        plt.text(V[att,i], V[att,j], attributeNames[att])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
    plt.title(titles[k] +'\n'+'Attribute coefficients')
    plt.axis('equal')

    # Plot cumulative variance explained
    plt.subplot(nrows, ncols,  5+k);
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')

plt.show()

c = np.array([])

for i in range(2):
    if i == 0:
        b1 = Y2*V[i]
    if i == 1:
        b2 = Y2*V[i]
plt.plot(b1,b2, 'x')

''' last part of ex2_1_5 to do'''
#last part of ex2_1_5  




