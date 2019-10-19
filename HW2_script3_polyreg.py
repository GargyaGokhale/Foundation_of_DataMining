# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:23:54 2019

Foundations of Data Mining
Polynomial Regression using Gradient Descent


@author: GargyaGokhale
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Data Acquisition"""
data = pd.read_excel('HW2data.xlsx')
grp_num = 11
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

X = np.append(x**2,x,axis=1)
X = np.append(X,np.ones((5,1)),axis=1)
Y = y

""" Initialise theta """
t2 = 0.007
t1 = -0.5
t0 = 7.2

""" Learning Rate """
alpha2 = 0.000001
alpha1 = 0.000001
alpha0 = 0.000001
count = 0

""" Gradient terms """
del_t2 = 1
del_t1 = 1
del_t0 = 1

""" Cost function stuff """
J = []
Xj = []

#for count in range(10000):
#while ((d1>0.001) or (d2>0.001)):
while ((abs(del_t1)>0.001 or abs(del_t0)>0.001 or abs(del_t2)>0.001) and count<10000) :
    """ h(x) = t2 x^2 + t1 x + t0 - y """
    z = t2 * (x**2) + t1*x + t0 - y
    z3 = z**3
    """ Cost Function Calculation """
    cost = np.mean((z**4))/4
    J.append(cost) 
    Xj.append(count)

    """ Calculate Gradient"""
    del_t2 = np.mean((z3)*(x**2))
    del_t1 = np.mean((z3)*x)
    del_t0 = np.mean(z3)
    
    """ Update theta """
    t2 = t2 - alpha2*del_t2
    t1 = t1 - alpha1*del_t1
    t0 = t0 - alpha0*del_t0
    
    count +=1
    """ Print first 5 iterations """
    if count<6:
        print("Iteration %d" % count)
        print("t2 is %f" % t2)
        print("t1 is %f" % t1)
        print("t0 is %f" % t0)
        print("****************************")
        

print("After Convergence")
print("Best fit quadratic is %f x2 + %f x + %f" % (t2,t1,t0))
print("Number of iterations = %d" % count)

"""Plots"""
theta = np.reshape(np.array([t2,t1,t0]),(3,1))
xx=np.reshape(np.sort(np.reshape(x,(1,5))),(5,1))
XX = np.append(xx**2,xx,axis=1)
XX = np.append(XX,np.ones((5,1)),axis=1)
Ycap = np.matmul(XX,theta)

plt.scatter(x,Y,marker='*',c='red')
plt.plot(xx,Ycap)
#plt.legend('Data','BestFit')
plt.xlabel("Temperature in Celcius")
plt.ylabel("Net Hourly Electrical Energy Output")
plt.show()

"""Cost Function Plot """
plt.plot(np.array(Xj),np.array(J))
plt.title("Cost Function vs Iterations")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost Function")
plt.show()

