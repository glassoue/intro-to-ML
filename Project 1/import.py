# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:02:38 2020

@author: james
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show, 
                               xlabel, ylabel, xticks, yticks, colorbar, cm, 
                               imshow, suptitle)

df = pd.read_excel(r"C:\Users\james\Documents\School\Machine Learning & Data mining\Project1\Data\hprice2.xls",header=None)

raw_data = df.get_values()
label= ['median housing price','crimes committed per capita','nitrous oxide, parts per 100 mill',
        'avg number of rooms per house','weighted dist. to 5 employ centers','accessibiliy index to radial hghwys',
        'property tax per $1000','average student-teacher ratio','% of people lower status','log(price)','log(nox)',
        'log(proptax)','log(crime)']

cols = range(0,12)

X = raw_data[:,cols]
#X[12] = np.log(X[1])

# mean and standard deviation of each attribute
mu = X.mean(axis=0)
s_ = X.std(axis=0)

# Seems normal to categorize 10 price ranges for houses
nbins = 10


for attribute in range(len(label)):
    for nbins in range(5):
        figure(figsize=(12,4))  
        title(str(label[attribute]))
        hist(X[:,attribute],bins=(nbins+1)*10)


# Plot scatter plot of data
#figure(figsize=(12,8))
#suptitle('2-D Normal distribution')
#
#subplot(1,2,1)
#plot(X[:,3], X[:,0], 'x')
#xlabel('x1'); ylabel('x2')
#title('Scatter plot of data')
#
#subplot(1,2,2)
#x = np.histogram2d(X[:,0], X[:,1], nbins)
#imshow(x[0], cmap=cm.gray_r, interpolation='None', origin='lower')
#colorbar()
#xlabel('x1'); ylabel('x2'); xticks([]); yticks([]);
#title('2D histogram')

