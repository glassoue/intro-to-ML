#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:06:56 2020

@author: ghassen97
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

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

K_outer = 5
K_inner = 5
CV_inner = model_selection.KFold(n_splits=K_inner,shuffle=True, random_state=111)
CV_outer = model_selection.KFold(n_splits=K_outer,shuffle=True, random_state=111)

n_hidden_units = [1,2,3,4,5]
errors_inner_ANN = np.empty((len(n_hidden_units),K_inner))
errors_outer_ANN = np.empty((K_outer,))
opt_hidden_units=[]
n_replicates = 1 
max_iter=5000

k1=0
for par_index, test_index in CV_outer.split(X,y):
    print('Outer cross-validation loop: {0}/{1}..'.format(k1+1,K_outer))
    print(40*'-')
       
    X_outer_train = torch.Tensor(X[par_index,:])
    y_outer_train = torch.Tensor(y[par_index])
    X_outer_test = torch.Tensor(X[test_index,:])
    y_outer_test = torch.Tensor(y[test_index])
    
    k2=0
    for train_index, val_index in CV_inner.split(X_outer_train,y_outer_train):
        print('Inner cross-validation loop: {0}/{1}..'.format(k2+1,K_inner))

        # Extract training and test set for current CV fold, convert to tensors
        X_inner_train = torch.Tensor(X[train_index,:])
        y_inner_train = torch.Tensor(y[train_index])
        X_inner_test = torch.Tensor(X[val_index,:])
        y_inner_test = torch.Tensor(y[val_index])
            
        for j, h in enumerate(n_hidden_units):
                        
            model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, h), #M features to n_hidden_units
                    torch.nn.ReLU(),   # 1st transfer function,
                    torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

            print('Training model of type:\n\n{}\n'.format(str(model())))
            
                # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_inner_train,
                                                       y=y_inner_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
            print('\n\tBest loss: {}\n'.format(final_loss))
            # Determine estimated class labels for test set
            y_test_est = net(X_inner_test) #not X_outer_test

            # Determine errors and errors
            se = (y_test_est.float()-y_inner_test.float())**2 # squared error
            MSE = (sum(se).type(torch.float)/len(y_inner_test)).data.numpy() #mean
            #errors.append(mse) # store error rate for current CV fold 
            errors_inner_ANN[j,k2] = MSE
            
            
        k2+=1

    
    # ANN Best model
    best_h = n_hidden_units[np.argmin(errors_inner_ANN.mean(1))]
    opt_hidden_units.append(best_h)
    print("ANN: Minimum error validation {} found for numbrer on units h={}".format(np.min(errors_inner_ANN.mean(1)), best_h))
    print()
    
        
    best_model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, best_h), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    print('Training model of type:\n\n{}\n'.format(str(model())))
    
        # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(best_model,
                                               loss_fn,
                                               X=X_outer_train,
                                               y=y_outer_train,
                                               n_replicates=n_replicates,
                                               max_iter=max_iter)

    print('\n\tBest loss: {}\n'.format(final_loss))
    # Determine estimated class labels for test set
    y_test_est_best = net(X_outer_test)
    
    # Determine errors and errors
    se = (y_test_est_best.float()-y_outer_test.float())**2 # squared error
    MSE = (sum(se).type(torch.float)/len(y_outer_test)).data.numpy() #mean
    #errors.append(mse) # store error rate for current CV fold 
    errors_outer_ANN[k1] = MSE

    k1+=1

#%%
'''I am looking forward to know the mistakes in this code'''




