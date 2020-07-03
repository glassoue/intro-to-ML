# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:38:36 2020

@author: james
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend, colorbar, imshow, xticks, yticks
from scipy.linalg import svd
from scipy.stats import zscore
from sklearn import tree
from platform import system
from os import getcwd
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from numpy import cov
from toolbox_02450 import rlr_validate
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from matplotlib.image import imread
from toolbox_02450 import windows_graphviz_call
from toolbox_02450 import feature_selector_lr, bmplot
import sklearn.linear_model as lm
from sklearn import model_selection

#%%
plt.style.use('ggplot')


#filename = r'C:\Users\james\Documents\School\Machine Learning & Data mining\Project1\Data\hprice2.csv'
#df = pd.read_csv(filename)



#Regression with all attributes for adiposity


from LoadData import *

# Load data from matlab file
#label = [name[0] for name in mat_data['label'][0]]

#label = ['crime','nox','rooms','dist','radial', 'proptax', 'stratio',  'lowstat']
#label = label[1:-4]

y,X = X[:,0], X[:,1:-4]
label = ['crime','nox','rooms','dist','radial', 'proptax', 'stratio',  'lowstat']


X = zscore(X)
#y = zscore(y)
N, M = X.shape

print(X.shape)
print(N)
print(M)

y2 = X - np.ones((N, 1))*X.mean(0)
y2 = y2*(1/np.std(y2,0))

model = lm.LinearRegression()
model.fit(y2,y)
# Predict adiposity
y_est1 = model.predict(y2)
residual = y_est1-y

#%%
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
#Gen_error = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
#Gen_error_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV.split(y2):
    
    # extract training and test set for current CV fold
    X_train = y2[train_index,:]
    y_train = y[train_index]
    X_test = y2[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]
    
    # Compute squared error with all features selected (no feature selection)
    
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    textout = '';
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    print('selected features for {} fold are: {}'.format(k+1,selected_features))
    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
        #Gen_error_fs[k] = Error_test_fs[k] - Error_train_fs[k] 
    figure(k)
    plt.subplot(1,2,1)
    plot(range(1,len(loss_record)), loss_record[1:])
    xlabel('Iteration')
    ylabel('Squared error (crossvalidation)')    
               
    plt.subplot(1,3,3)
    bmplot(label, range(1,features_record.shape[1]), -features_record[:,1:])
    plt.clim(-1.5,0)
    xlabel('Iteration')
    

    k+=1


print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
#print('- Generalization error:   {0}'.format(Gen_error.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
#print('- Generalization error:   {0}'.format(Gen_error_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))
#%%
figure(k)
plt.subplot(1,3,2)
bmplot(label, range(1,Features.shape[1]+1), -Features)
plt.clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')
plt.savefig('Kfold_selected_features.png')


f=2 # cross-validation fold to inspect
ff=Features[:,f-1].nonzero()[0]

model = lm.LinearRegression()
model.fit(y2[:,ff],y)
# Predict adiposity
y_est1 = model.predict(y2[:,ff])
residual = y_est1-y



# residual errors
figure(figsize=(8,6))
plot(X[:,ff],residual,'.')
ylabel('residual error')

# Display scatter plot
figure(figsize=(12,6))
plt.subplot(2,2,1)
#plot(X, y, '.')
plot(y, y_est1, '.')
xlabel('Price (true)'); ylabel('Price (estimated)');
#legend(['Training data', 'Regression fit (model)'])
plt.subplot(2,2,2)
plt.hist(residual,40) #error plot
plt.savefig('price_after_selection.png')
show()
#%%
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
label = [u'Offset']+label
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
#Gen_error = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
#Gen_error_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))
#%%
k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    #Gen_error[k] = Error_test[k] - Error_train[k]
    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(10,5))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()

        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
        plt.savefig('reg.png')

    k+=1
#%%
show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
#print('- Generalization error:   {0}'.format(Gen_error.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
#print('- Generalization error:   {0}'.format(Gen_error_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
arr = np.empty((M))
for m in range(M):
    print('{:>15} {:>15}'.format(label[m], np.round(w_rlr[m,-1],2)))
    arr[m] = np.round(w_rlr[m,-1],2)
    
print(arr[1:])

print(label[1:])


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
y_pos = np.arange(len(label))
#performance = 3 + 10 * np.random.rand(len(people))
#error = np.random.rand(len(label[1:]))

ax.barh(y_pos[0:-1], arr, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(label)
ax.invert_yaxis()  # labels read top-to-bottom
plt.savefig('weight_graph.png')
plt.show()