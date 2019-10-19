# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:02:30 2019

Foundations of Data Mining
Linear Regression using Gradient Descent

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
transformation = True

""" Min-Max Scalling """
if transformation == True:
    anum = np.min(y)
    den = np.max(y) - anum
    y = (y - anum)/den

X = np.append(x,np.ones((5,1)),axis=1)
Y = y

""" Initialise theta """
t1 = 0.
t0 = 1.

""" Learning Rate """
alpha1 = 0.001
alpha0 = 0.001
count = 0

""" Gradient terms """
del_t1 = 1
del_t0 = 1

""" Cost function stuff """
J = []
Xj = []

#for count in range(10000):
#while ((d1>0.001) or (d2>0.001)):
while ((abs(del_t1)>0.01 or abs(del_t0)>0.01) and count<10000) :

    """ Cost Function Calculation """
    cost = np.mean((t1*x + t0 - y)**2)/2
    J.append(cost) 
    Xj.append(count)
    
    """ Calculate Gradient"""
    del_t1 = np.mean((t1*x + t0 - y)*x)
    del_t0 = np.mean(t1*x + t0 - y)
    
    """ Update theta """
    t1 = t1 - alpha1*del_t1
    t0 = t0 - alpha0*del_t0
    
    count +=1
    """ Print first 5 iterations """
    if count<6:
        print("Iteration %d" % count)
        print("Slope is %f" % t1)
        print("Y-intercept is %f" % t0)
        print("Delta theta1 is %f" % del_t1)
        print("Delta theta0 is %f" % del_t0)
        print("****************************")
        


print("After Convergence")
print("Best fit line is %f x + %f" % (t1,t0))
print("Number of iterations = %d" % count)

"""Plots"""

theta = np.reshape(np.array([t1,t0]),(2,1))
xx = np.reshape(np.sort(np.reshape(x,(1,5))),(5,1))
XX = np.append(xx,np.ones((5,1)),axis=1)
Ycap = np.matmul(XX,theta)

""" Actual Curves """
plt.scatter(x,Y,marker='*',c='red')
plt.plot(xx,Ycap)
#plt.legend('Data','BestFit')
plt.title("Electrical Energy vs Temperature")
plt.xlabel("Temperature in Celcius")
plt.ylabel("Net Hourly Electrical Energy Output")
plt.show()

"""Cost Function Plot """
plt.plot(np.array(Xj),np.array(J))
plt.title("Cost Function vs Iterations")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost Function")
plt.show()

