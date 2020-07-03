
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats


from sklearn.model_selection import train_test_split


##### importing sklearn modules
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.utils import class_weight

from sklearn import model_selection, tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

#importing matplot modules
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend




filename = 'hprice2.csv'
df = pd.read_csv(filename)
# predicting crime rate , so we are converting lcrime greater than zero to high and less than zero to low

TARGET_Y = 'radial' # change to lcrime if you want perform classification on lcrime data
drop_str = ['lprice','lnox','lproptax','lcrime']

if TARGET_Y=='lcrime':
    # predicting crime rate , so we are converting lcrime greater than zero to high and less than zero to low
    df.loc[df.lcrime > 0, 'lcrime'] = 1

    df.loc[df.lcrime <= 0, 'lcrime'] = 0
    drop_str = ['lprice','lnox','lproptax']






# Preparing x and y inputs
droped_df = df.drop(drop_str,axis=1)

X = droped_df.drop(str(TARGET_Y), axis=1).values

X=stats.zscore(X)


y = df[str(TARGET_Y)].values

print(X.shape)
print(y.shape)

#Selecting range of parameters

lamd_intervl=np.logspace(-16, 4, 50)

neighbors = np.arange(1, 45, 1)

depth = np.arange(2, 51, 1)

#Initializing fold

kf1 = KFold(10,shuffle=True)
kf2 = KFold(10,shuffle=True)
error_metrics1=np.empty((10,))
error_metrics2=np.empty((10,))
error_metrics3=np.empty((10,))
error_metrics4=np.empty((10,))

error_model1=np.empty((len(lamd_intervl),10))
error_model2=np.empty((len(neighbors),10))
error_model3=np.empty((len(depth),10))
error_model4=[]


suited_lamdas=[]
selected_depths = []
best_neighbours = []



    
    
#outer fold

kfold1=0
for train_index, test_index in kf1.split(X):
    X_partial, X_test_partial = X[train_index], X[test_index]
    y_train_partial, y_test_partial = y[train_index], y[test_index]
    actual1 = y_test_partial
    kfold2=0
    # inner fold
    for train_index, test_index in kf2.split(X_partial):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        actual = y_test
#logistic regression
        for i, j in enumerate(lamd_intervl):
            model1 = LogisticRegression(penalty='l2',C=1/lamd_intervl[i], solver='liblinear')
            model1.fit(X_train, y_train)
            y_pred1 = model1.predict(X_test)

            
            error1 = np.sum(y_pred1 != actual) / float(len(actual))
      
            error_model1[i,kfold2]=error1
# KNN
        for k,l in enumerate(neighbors):
            model2 = KNeighborsClassifier(n_neighbors=l,weights='distance',metric='cosine')
            model2.fit(X_train, y_train)
            y_pred2 = model2.predict(X_test)

            error2 = np.sum(y_pred2 != actual) / float(len(actual))
       
            error_model2[k,kfold2]=error2
# decision tree
        for c, d in enumerate(depth):
            model3 = tree.DecisionTreeClassifier(criterion='gini',max_depth=d)
            model3 = model3.fit(X_train,y_train)
            y_pred3 = model3.predict(X_test)

            error3 = np.sum(y_pred3 != actual) / float(len(actual))
       

            error_model3[c-1,kfold2]=error3

        kfold2=kfold2+1



    best_lambda = lamd_intervl[np.argmin(error_model1.mean(1))]

    suited_lamdas.append(best_lambda)
    print("Logi Regrsn: Min err {} value for lamda = {}".format(np.min(error_model1.mean(1)),best_lambda))
    print()
    
    best_model1 = LogisticRegression(penalty='l2', C=1/best_lambda)
    best_model1 = best_model1.fit(X_partial,y_train_partial)
    y_est_test_model1 = best_model1.predict(X_test_partial)
    
    test_model1 = np.sum(y_est_test_model1 != y_test_partial) / float(len(y_est_test_model1))
    error_metrics1[kfold1] = test_model1

    #

    # Best K-neighbors classifier
    suited_k = neighbors[np.argmin(error_model2.mean(1))]
    best_neighbours.append(suited_k)
    print("K-NN : Min err {} value for n ={}".format(np.min(error_model2.mean(1)), suited_k))
    print()
    
    best_model2 = KNeighborsClassifier(n_neighbors=suited_k, metric ='cosine')
    best_model2 = best_model2.fit(X_partial,y_train_partial)
    y_est_test_model2 = best_model2.predict(X_test_partial)
    
    test_model2 = np.sum(y_est_test_model2 != y_test_partial) / float(len(y_est_test_model2))
    error_metrics2[kfold1] = test_model2


     # Decision Tree Best model
    suited_depth = depth[np.argmin(error_model3.mean(1))]
    selected_depths.append(suited_depth)
    print("Des Tree: Min err {} value for depth ={}".format(np.min(error_model3.mean(1)), suited_depth))
    print()
    best_model3 = tree.DecisionTreeClassifier(criterion='gini', max_depth=suited_depth)
    best_model3 = best_model3.fit(X_partial,y_train_partial)
    y_est_test_model3 = best_model3.predict(X_test)
    
    
    test_model3 = np.sum(y_est_test_model3 != y_test) / float(len(y_est_test_model3))
    error_metrics3[kfold1] = test_model3



    #Baseline 
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model = baseline_model.fit(X_partial,y_train_partial)
    y_pred_test_baseline = baseline_model.predict(X_test)
    
    misclass_rate_test_baseline = np.sum(y_pred_test_baseline != y_test) / float(len(y_pred_test_baseline))
    error_metrics4[kfold1]= misclass_rate_test_baseline
    print("Baseline error {}".format(misclass_rate_test_baseline))
    print()
    print(70*'-')


    if kfold1+1==10:
        f = figure()
        plot(lamd_intervl, error_model1.mean(1),color='green')
        title(str(TARGET_Y)+'_Attribute Optimised Log Regrssn Model {}th outer fold'.format(kfold1+1))
        xlabel('lambda')
        ylabel('Val Error K Fold=10 inner fld'.format(kfold2)) 
        plt.scatter(best_lambda, np.min(error_model1.mean(1)), s=190, marker='x', color='r', label = 'Min Val Error')
        plt.savefig(str(TARGET_Y)+'_Regrssn_2_level_cross_val.png')
        plt.show()
  

        f = figure()
        plot(neighbors, error_model2.mean(1),color='green')
        title(str(TARGET_Y)+'_Attribute Optimised KNN Model for the {}th outer fold'.format(kfold1+1))
        xlabel('Number of neighbors')
        ylabel('Val Error K Fold=10 inner fld'.format(kfold2))
        plt.scatter(suited_k, np.min(error_model2.mean(1)), s=190, marker='x', color='r', label = 'Min Val Error')
        plt.savefig(str(TARGET_Y)+'_KNN_2_level_cross_val.png')
        plt.show()

        

        f = figure()
        plot(depth, error_model3.mean(1),color='green')
    

        title(str(TARGET_Y)+'_Attribute Optimised Decision Tree for the {}th outer fold'.format(kfold1+1))
        xlabel('Max depth')
        ylabel('Val Error K Fold=10 inner fld'.format(kfold2)) 
        plt.scatter(suited_depth, np.min(error_model3.mean(1)), s=190, marker='x', color='r', label = 'Min Val Error')
        plt.savefig(str(TARGET_Y)+'_Decision_tree_2_level_cross_val.png')
        plt.show()
        legend()

    kfold1+=1






print(suited_lamdas)
print(error_metrics1)
print()

print(selected_depths)
print(error_metrics2)
print()

print(best_neighbours)
print(error_metrics3)
print()

print("Basiline")
print(error_metrics4)





