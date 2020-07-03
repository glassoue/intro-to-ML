#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:35:52 2020

@author: ghassen97
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.linalg import svd

from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy import stats
import scipy.stats as st
#%%
'''load data '''
df = pd.read_excel(r"/home/ghassen97/Desktop/S8/Intro to ML/project/dataset/hprice2.xls",header=None)

raw_data = df.to_numpy() 

#cols = range(0,9) 
X = raw_data[:,1:9 ]
N,M= X.shape
y=raw_data[:,0]
y=np.reshape(y,(len(y),1))
attributeNames=['crime','nox','rooms','dist','radial', 'proptax', 'stratio',  'lowstat']

# Normalize data
X = stats.zscore(X) #standardized data

# K-fold crossvalidation
K_outer = 10
K_inner = 10
CV_inner = model_selection.KFold(n_splits=K_inner,shuffle=True, random_state=111)
CV_outer = model_selection.KFold(n_splits=K_outer,shuffle=True, random_state=111)

# Parameters for ridge regression
#lambda_interval = np.power(10.,range(-5,9))  ##to change
lambda_interval=np.linspace(-10, 100,110)  
errors_inner_ridge = np.empty((len(lambda_interval),K_inner))
errors_outer_ridge = np.empty((K_outer,))  #errors_outer_ridge = errors_test_ridge
opt_lambdas=[]

# Parameters for neural network classifier
#n_hidden_units =np.arange(1,25 ).tolist() # [1,2,4,3,5,6,7]
n_hidden_units=[1,2,3,4,5,6,7]
errors_inner_ANN = np.empty((len(n_hidden_units),K_inner))
errors_outer_ANN = np.empty((K_outer,))
opt_hidden_units=[]

#Baseline , need only outer CV
errors_outer_baseline = np.empty((K_outer,))

y_test_all = []
y_hat_baseline_all= [] 
y_hat_ridge_all= []
y_hat_ANN_all= []

k_outer=0
for train_outer_index, test_outer_index in CV_outer.split(X):
    print('Outer cross-validation loop: {0}/{1}..'.format(k_outer+1,K_outer))
    print(40*'-')
    
    X_outer_train= X[train_outer_index,:] 
    y_outer_train= y[train_outer_index] 
    X_outer_test = X[test_outer_index,:]
    y_outer_test = y[test_outer_index]
        
    k_inner=0
    for train_inner_index, test_inner_index in CV_inner.split(X_outer_train):
        print('Inner cross-validation loop: {0}/{1}..'.format(k_inner+1,K_inner))

        X_inner_train= X[train_inner_index,:]
        y_inner_train= y[train_inner_index]
        X_inner_test = X[test_inner_index,:]
        y_inner_test = y[test_inner_index]

        for i, k in enumerate(lambda_interval):
            ridge1 = Ridge(alpha=k,fit_intercept=True)
            ridge1.fit(X_inner_train, y_inner_train)
            y_hat_inner_ridge1 = ridge1.predict(X_inner_test)

            errors_inner_ridge[i,k_inner]=mse(y_inner_test, y_hat_inner_ridge1)
        
        for j, h in enumerate(n_hidden_units):
            ANN = MLPRegressor(hidden_layer_sizes=(h,), max_iter=5000) ## max_iter to change
            ANN = ANN.fit(X_inner_train,y_inner_train)
            y_hat_inner_ANN = ANN.predict(X_inner_test)

            errors_inner_ANN[j,k_inner] = mse(y_inner_test, y_hat_inner_ANN)
            
        k_inner+=1
    
    # best model of Ridge Regression
    print()
    opt_lambda = lambda_interval[np.argmin(errors_inner_ridge.mean(1))]
    opt_lambdas.append(opt_lambda)
    print("Ridge Regression: Minimum inner error {} found for lambda= {}".format(np.min(errors_inner_ridge.mean(1)),opt_lambda))
    print()
    
    opt_ridge = Ridge(alpha=float(opt_lambda))
    opt_ridge = opt_ridge.fit(X_outer_train,y_outer_train)
    y_hat_outer_test_opt_ridge = opt_ridge.predict(X_outer_test)
     
    errors_outer_ridge[k_outer] = mse(y_outer_test, y_hat_outer_test_opt_ridge)
    
    # best model of ANN
    opt_h = n_hidden_units[np.argmin(errors_inner_ANN.mean(1))]
    opt_hidden_units.append(opt_h)
    print("ANN: Minimum error inner {} found for numbrer on units h={}".format(np.min(errors_inner_ANN.mean(1)), opt_h))
    print()
    opt_ANN = MLPRegressor(hidden_layer_sizes=(opt_h,), max_iter=5000)
    opt_ANN = opt_ANN.fit(X_outer_train,y_outer_train)
    y_hat_outer_test_opt_ANN = opt_ANN.predict(X_outer_test)
    
    errors_outer_ANN[k_outer] = mse(y_outer_test, y_hat_outer_test_opt_ANN)
    
    #Baseline 
    baseline = DummyRegressor()
    baseline = baseline.fit(X_outer_train,y_outer_train)
    y_hat_outer_test_baseline = baseline.predict(X_outer_test)
    
    #mse_test_baseline = mse(y_outer_test, y_hat_outer_test_baseline)
    errors_outer_baseline[k_outer] = mse(y_outer_test, y_hat_outer_test_baseline)  #mse_test_baseline
    print("Baseline error {}".format(mse(y_outer_test, y_hat_outer_test_baseline)))
    print()
    print(40*'-')
    
    
    y_test_all.append(y_outer_test) 
    y_hat_ridge_all.append(y_hat_outer_test_opt_ridge)
    y_hat_ANN_all.append(y_hat_outer_test_opt_ANN)
    y_hat_baseline_all.append(y_hat_outer_test_baseline)
    
    k_outer+=1
 
#y_test_all is a list, concatenate it to array
y_test_all        =np.concatenate(y_test_all)  
y_hat_ridge_all   =np.concatenate(y_hat_ridge_all)   
y_hat_ANN_all     =np.concatenate(y_hat_ANN_all) 
y_hat_baseline_all=np.concatenate(y_hat_baseline_all) 

y_hat_ridge_all=np.reshape(y_hat_ridge_all,(len(y_hat_ridge_all),1))
y_hat_ANN_all=np.reshape(y_hat_ANN_all,(len(y_hat_ANN_all),1)) #shape was (506,),now changed to (506,1)
y_hat_baseline_all=np.reshape(y_hat_baseline_all,(len(y_hat_baseline_all),1))


print("Ridge")
print(opt_lambdas)

print(np.around(np.array(errors_outer_ridge),4))  ##to change
print()

print("ANN")
print(opt_hidden_units)

print(np.around(np.array(errors_outer_ANN),4))
print()


print("Basiline")

print(np.around(np.array(errors_outer_baseline),4))
#%%
"""Comparing Models """
"""
H0: model A and model B have the same performance
H1: model A and model B do not the same performance

z=zA-zB

p<0.05 : the lower p is the more evidence there is that A is better then B, 
         so reject H0 and take H1
"""
alpha=0.05
zRidge   =np.abs(y_test_all - y_hat_ridge_all ) ** 2 
zANN     =np.abs(y_test_all - y_hat_ANN_all ) ** 2 
zBaseline=np.abs(y_test_all - y_hat_baseline_all ) ** 2 

# Ridge regression and ANN
# Compute confidence interval of z1 = zRidge - zANN and p-value of Null hypothesis
#z1=zRidge - zANN
z1=zANN - zRidge

CI1 = st.t.interval(1-alpha, len(z1)-1, loc=np.mean(z1), scale=st.sem(z1)) 
p1 = st.t.cdf( -np.abs( np.mean(z1) )/st.sem(z1), df=len(z1)-1)  # p-value


# Ridge regression and Baseline 
# Compute confidence interval of z2 = zRidge - zBaseline and p-value of Null hypothesis
z2 = zRidge - zBaseline
CI2 = st.t.interval(1-alpha, len(z2)-1, loc=np.mean(z2), scale=st.sem(z2)) 
p2 = st.t.cdf( -np.abs( np.mean(z2) )/st.sem(z2), df=len(z2)-1)  # p-value


# ANN and Baseline
# Compute confidence interval of z3 = zBaseline - zANN and p-value of Null hypothesis
z3 = zBaseline - zANN
CI3 = st.t.interval(1-alpha, len(z3)-1, loc=np.mean(z3), scale=st.sem(z3)) 
p3 = st.t.cdf( -np.abs( np.mean(z3) )/st.sem(z3), df=len(z3)-1)  # p-value

# # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
# zB = np.abs(y_test - yhatB ) ** 2
# z = zA - zB
# CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
# p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value




