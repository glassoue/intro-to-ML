# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:45:22 2020

@author: james
"""

# exercise 4.2.1

import numpy as np
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook(r'C:\Users\james\Documents\School\Machine Learning & Data mining\Project1\Data\hprice2_test.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(0,0,11)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(11,1,506)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

# Extract vector y, convert to NumPy matrix and transpose
y = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract data to matrix X
X = np.empty((505,11))
for i in range(11):
    X[:,i] = np.array(doc.col_values(i,1,506)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

print('Ran NamedAttributes')