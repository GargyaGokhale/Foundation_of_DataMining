# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:00:39 2019
HW3 Data Visualisation
@author: GargyaGokhale
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_excel("HW3Atrain.xlsx")

y0=[]
y1=[]
length = len(data.y)

for i in range(length):
    if data.y[i] == 0:
        y0.append(i)
    elif data.y[i] == 1:
        y1.append(i)

X0_r = []
X1_r = []

for i in y0:
    X0_r.append(data.X_0[i])
    X1_r.append(data.X_1[i])

X0_b = []
X1_b = []

for i in y1:
    X0_b.append(data.X_0[i])
    X1_b.append(data.X_1[i])


plt.scatter(X0_r,X1_r,color='red')
plt.scatter(X0_b,X1_b,color='blue')
plt.legend({"Label_0","Label_1"})
plt.show()