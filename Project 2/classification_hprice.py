import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

##### importing sklearn modules

from sklearn.model_selection import train_test_split


from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import cross_val_score

#####
from sklearn import model_selection, tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier



filename = 'hprice2.csv'
df = pd.read_csv(filename)
TARGET_Y = 'lcrime' # change to lcrime if you want perform classification on lcrime data
drop_str = ['lprice','lnox','lproptax','lcrime']

# assigning values obtained from two level cross validation

best_lamda=1e-08
best_depth=48
best_neighbour=3


if TARGET_Y=='lcrime':
    # predicting crime rate , so we are converting lcrime greater than zero to high and less than zero to low
    df.loc[df.lcrime > 0, 'lcrime'] = 1

    df.loc[df.lcrime <= 0, 'lcrime'] = 0
    drop_str = ['lprice','lnox','lproptax']

    best_lamda=0.0007906043210907702
    best_depth=2
    best_neighbour=4



# Preparing x and y inputs

droped_df = df.drop(drop_str,axis=1)

X = droped_df.drop(str(TARGET_Y), axis=1).values

X=stats.zscore(X)


y = df[str(TARGET_Y)].values



print(X.shape)
print(y.shape)




kf = KFold(10,shuffle=True)
error_metrics1=[]
error_metrics2=[]
error_metrics3=[]
error_metrics4=[]
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	actual = y_test


	model1 = LogisticRegression(penalty='l2',C=1/0.0007906043210907702, solver='liblinear')
	model1.fit(X_train, y_train)
	y_pred1 = model1.predict(X_test)

	
	error1 = np.sum(y_pred1 != actual) / float(len(actual))
	error_metrics1=np.append(error_metrics1,error1)



	model2 = KNeighborsClassifier(n_neighbors=4,weights='distance',metric='cosine')
	model2.fit(X_train, y_train)
	y_pred2 = model2.predict(X_test)

	error2 = np.sum(y_pred2 != actual) / float(len(actual))
	error_metrics2=np.append(error_metrics2,error2)



	model3 = tree.DecisionTreeClassifier(criterion='gini',max_depth=2)
	model3 = model3.fit(X_train,y_train.ravel())
	y_pred3 = model3.predict(X_test)

	error3 = np.sum(y_pred3 != actual) / float(len(actual))
	error_metrics3=np.append(error_metrics3,error3)

	baseline = DummyClassifier(strategy="most_frequent")
	baseline = baseline.fit(X_train,y_train)
	y_pred4 = baseline.predict(X_test)

	error4 = np.sum(y_pred4 != actual) / float(len(actual))
	error_metrics4=np.append(error_metrics4,error4)


results = [error_metrics1, error_metrics2, error_metrics3,error_metrics4]


print(results)
names = ['Logistic Regression', 'K-neighbors', 'Decision Tree','baseline']

# fig.suptitle('Algorithm Comparison using K1=10 outer folds')

fig, ax = plt.subplots(figsize=(8, 6))

ax.set_title(str(TARGET_Y)+'_classification error in each fold')
th = np.linspace(0, 10,10)



ax.plot(th,error_metrics1,label='logistic regression')
ax.plot(th,error_metrics2,label='KNN')
ax.plot(th,error_metrics3,label='decision tree')
ax.plot(th,error_metrics4,label='baseline')
# ax.set_xticklabels(names)
# ax.set_ylabel("Test Error (misclassification erro rate %)")
plt.legend()
plt.xlabel('fold', fontsize=18)
plt.ylabel('error', fontsize=18)
plt.savefig('crime_error_comparison_class.png')
plt.show()

fig1 = plt.figure(figsize=(8, 6))

fig1.suptitle('Algorithm Comparison,total 10 fold, outer folds_'+str(TARGET_Y))

ax1 = fig1.add_subplot(111)
ax1.set_xticklabels(names)
ax1.set_ylabel("Test Error (MSE)")
plt.boxplot(results)

plt.savefig('alg_comparison_class_crime.png')
plt.show()

