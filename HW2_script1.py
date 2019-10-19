# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:35:30 2019

Foundations of Data Mining
Linear Regression 

@author: GargyaGokhale
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Data Acquisition"""
data = pd.read_excel('HW2data.xlsx')
grp_num = 2
n = ((grp_num - 1)*5 +  2) - 2

x = np.reshape((np.array(data.X[n:n+5])),(5,1))
y = np.reshape((np.array(data.Y[n:n+5])),(5,1))

"""Data Transformation"""
transformation = False

if transformation == True:
    anum = np.min(y)
    den = np.max(y) - anum
    y = (y - anum)/den

"""Closed form solution of Least Squares"""
X = np.append(x,np.ones((5,1)),axis=1)
Y = y
temp = np.linalg.inv(np.matmul(np.transpose(X),X))
theta = np.matmul(np.matmul(temp,np.transpose(X)),Y)


print("Solution is \n")
print(theta)
print("Best fit line is %f x + %f" %(theta[0],theta[1]))

"""Plots"""
xx = np.reshape(np.sort(np.reshape(x,(1,5))),(5,1))
XX = np.append(xx,np.ones((5,1)),axis=1)
Ycap = np.matmul(XX,theta)

plt.scatter(x,Y,marker='*',c='red')
plt.plot(xx,Ycap)
#plt.legend('Data','BestFit')
plt.xlabel("Temperature in Celcius")
plt.ylabel("Net Hourly Electrical Energy Output")
plt.show()

