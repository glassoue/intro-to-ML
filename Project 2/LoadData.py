# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:56:04 2020

@author: james
"""

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

df = pd.read_excel(r"C:\Users\james\Documents\School\Machine Learning & Data mining\Project2\Data\hprice2.xls",header=None)

raw_data = df.get_values()
label= ['median housing price','crimes committed per capita','nitrous oxide, parts per 100 mill',
        'avg number of rooms per house','weighted dist. to 5 employ centers','accessibiliy index to radial hghwys',
        'property tax per $1000','average student-teacher ratio','% of people lower status','log(price)','log(nox)',
        'log(proptax)','log(crime)']

cols = range(0,12)

X = raw_data[:,cols]