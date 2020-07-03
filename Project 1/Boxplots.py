# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:54:04 2020

@author: james
"""

# Exercise 4.2.4
from matplotlib.pyplot import (figure, subplot, boxplot, title, xticks, ylim, 
                               show, tight_layout)
import numpy as np
# requires data from exercise 4.1.1
from NamedAttributes import *


X[:,1] = np.log10(X[:,1])


figure(figsize=(15,8))
for m in range(M):
    subplot(1,M,m+1)
    boxplot(X[:,m])
    xlabel(attributeNames[m])
    tight_layout()