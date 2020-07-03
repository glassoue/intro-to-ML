"""
Created on Sat Feb 22 14:02:38 2020

@author: james
"""
import numpy as np
import pandas as pd

from matplotlib.pyplot import (figure, title, subplot, plot, hist, show, 
                               xlabel, ylabel, xticks, yticks, colorbar, cm, 
                               imshow, suptitle,savefig)






#X[12] = np.log(X[1])
nbins = 10

df = pd.read_excel(r"D:\machine_learning_files\data_for_report\hprice2.xls",header=None)
cols = range(0,12)
raw_data = df.get_values()
X = raw_data[:,cols]

label= ['median housing price','crimes committed per capita','nitrous oxide, parts per 100 mill',
        'avg number of rooms per house','weighted dist. to 5 employ centers','accessibiliy index to radial hghwys',
        'property tax per $1000','average student-teacher ratio','% of people lower status','log(price)','log(nox)',
        'log(proptax)']



for i in range(len(label)):
    print (label[i])
    figure(figsize=(12,4))
    title(str(label[i]))
    a=hist(X[:,i],bins=nbins)
    savefig(str(label[i]+'.png'))
# mean and standard deviation of each attribute
mu = X.mean(axis=0)
s_ = X.std(axis=0)

# Seems normal to categorize 10 price ranges for houses







# Plot scatter plot of data
figure(figsize=(12,8))
suptitle('2-D Normal distribution')

subplot(1,2,1)
plot(X[:,0], X[:,1], 'x')
xlabel('x1'); ylabel('x2')
title('Scatter plot of data')

subplot(1,2,2)
x = np.histogram2d(X[:,0], X[:,1], nbins)
imshow(x[0], cmap=cm.gray_r, interpolation='None', origin='lower')
colorbar()
xlabel('x1'); ylabel('x2'); xticks([]); yticks([]);
title('2D histogram')
