# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:54:01 2019
Test Plot
@author: GargyaGokhale
"""
import numpy as np
import matplotlib.pyplot as plt
a=[13.97,22.1,6.7]
x=[]
y=[]
for i in a:
    x.append(i)
    y.append(0.004*i*i - 0.058*i + 1.408)
    
X = np.array(x)
Y = np.array(y)

plt.plot(X,Y)
plt.show()
