# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:23:29 2020

@author: james
"""

import matplotlib.pyplot as plt
import scipy
import scipy.stats

df = pd.read_excel(r"C:\Users\james\Documents\School\Machine Learning & Data mining\Project1\Data\hprice2.xls",header=None)
cols = range(0,11)
raw_data = df.get_values()
X = raw_data[:,cols]

size = 30000
x = scipy.arange(size)
y = scipy.int_(scipy.round_(scipy.stats.vonmises.rvs(5,size=size)*47))
h = plt.hist(y, bins=range(48))

dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']

for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(y)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlim(0,47)
plt.legend(loc='upper right')
plt.show()