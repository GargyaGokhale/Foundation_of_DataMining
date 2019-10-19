# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:58:07 2019

Basic Code for MLP Implementation from Scratch
HW3A

@author: GargyaGokhale
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

""" Function Area"""

def reLu(X):
    return np.maximum(X,0)

def delReLu(X):
    return 1*(X>0)

def sigmoid(X):
    return 1/(1+np.exp(-X))

def delSigmoid(X):
    return sigmoid(X)*(1-sigmoid(X))

def crossEntropy(O,Y):
    Z = -1*(Y*np.log(O + np.exp(-15)) + (1-Y)*np.log(1-O + np.exp(-15)))
    return Z

def delCrossEntropy(O,Y):
    Z = ((1-Y)/(1-O)) - (Y/O)
    return Z
    

""" Main Script"""

data = pd.read_excel("HW3Atrain.xlsx")

""" Initalise NN, set random values to weights """
np.random.seed(1)
a=0.05

Wih = a*np.random.rand(2,10)
bih = np.random.rand(1,10)

Whh = a*np.random.rand(10,10)
bhh = np.random.rand(1,10)

Who = a*np.random.rand(10,1)
bho = np.random.rand(1,1)


#""" Set batches """
#seq = np.random.permutation(len(data.y))
n =45

X0 = np.reshape(np.array(data.X_0[0:n]),(1,n))
X1 = np.reshape(np.array(data.X_1[0:n]),(1,n))
Y = np.reshape(np.array(data.y[0:n]),(n,1))
X = np.append(X0,X1,axis=0)
J=[]

alpha = 0.0001

iterations = 10000
for it in range(iterations):
    
    """ Feed Forward """
    Zih = np.dot(np.transpose(X),Wih) + bih
    Oih = np.transpose(reLu(Zih))
    
    Zhh = np.dot(np.transpose(Oih),Whh) + bhh
    Ohh = np.transpose(reLu(Zhh))
    
    Zho = np.dot(np.transpose(Ohh),Who) + bho
    O = sigmoid(Zho)
    
    """ Evaluate Cost """
    
    j = np.mean(crossEntropy(O,Y))
    J.append(j)
    
    """ Calculate Errors and Gradients """
    err_oh = delCrossEntropy(O,Y)*delSigmoid(Zho)/n
    grad_ho = np.dot(Ohh,err_oh)
    grad_bho = np.sum(err_oh,axis=0)
    
    err_hh = np.dot(err_oh,np.transpose(Who))*delReLu(Zhh)
    grad_hh = np.dot(Oih,err_hh)
    grad_bhh = np.sum(err_hh,axis=0)
    
    err_hi = np.dot(err_hh,np.transpose(Whh))*delReLu(Zih)
    grad_ih = np.dot(X,err_hi)
    grad_bih = np.sum(err_hi,axis=0)
    
    """ Update Weights """
    Who -= alpha*grad_ho
    bho -= alpha*grad_bho
    Whh -= alpha*grad_hh
    bhh -= alpha*grad_bhh
    Wih -= alpha*grad_ih
    bih -= alpha*grad_bih

    
print("Cost Function after %d iterations is %f" % (iterations,j))    

""" Batch 2 """

X0 = np.reshape(np.array(data.X_0[n:2*n]),(1,n))
X1 = np.reshape(np.array(data.X_1[n:2*n]),(1,n))
Y = np.reshape(np.array(data.y[n:2*n]),(n,1))
X = np.append(X0,X1,axis=0)
#J=[]

alpha = 0.0001

iterations = 10000
for it in range(iterations):
    
    """ Feed Forward """
    Zih = np.dot(np.transpose(X),Wih) + bih
    Oih = np.transpose(reLu(Zih))
    
    Zhh = np.dot(np.transpose(Oih),Whh) + bhh
    Ohh = np.transpose(reLu(Zhh))
    
    Zho = np.dot(np.transpose(Ohh),Who) + bho
    O = sigmoid(Zho)
    
    """ Evaluate Cost """
    
    j = np.mean(crossEntropy(O,Y))
    J.append(j)
    
    """ Calculate Errors and Gradients """
    err_oh = delCrossEntropy(O,Y)*delSigmoid(Zho)/n
    grad_ho = np.dot(Ohh,err_oh)
    grad_bho = np.sum(err_oh,axis=0)
    
    err_hh = np.dot(err_oh,np.transpose(Who))*delReLu(Zhh)
    grad_hh = np.dot(Oih,err_hh)
    grad_bhh = np.sum(err_hh,axis=0)
    
    err_hi = np.dot(err_hh,np.transpose(Whh))*delReLu(Zih)
    grad_ih = np.dot(X,err_hi)
    grad_bih = np.sum(err_hi,axis=0)
    
    """ Update Weights """
    Who -= alpha*grad_ho
    bho -= alpha*grad_bho
    Whh -= alpha*grad_hh
    bhh -= alpha*grad_bhh
    Wih -= alpha*grad_ih
    bih -= alpha*grad_bih

    
print("Cost Function after %d iterations is %f" % (iterations,j))    

""" Batch 3 """

X0 = np.reshape(np.array(data.X_0[2*n:3*n]),(1,n))
X1 = np.reshape(np.array(data.X_1[2*n:3*n]),(1,n))
Y = np.reshape(np.array(data.y[2*n:3*n]),(n,1))
X = np.append(X0,X1,axis=0)
#J=[]

alpha = 0.0001

iterations = 10000
for it in range(iterations):
    
    """ Feed Forward """
    Zih = np.dot(np.transpose(X),Wih) + bih
    Oih = np.transpose(reLu(Zih))
    
    Zhh = np.dot(np.transpose(Oih),Whh) + bhh
    Ohh = np.transpose(reLu(Zhh))
    
    Zho = np.dot(np.transpose(Ohh),Who) + bho
    O = sigmoid(Zho)
    
    """ Evaluate Cost """
    
    j = np.mean(crossEntropy(O,Y))
    J.append(j)
    
    """ Calculate Errors and Gradients """
    err_oh = delCrossEntropy(O,Y)*delSigmoid(Zho)/n
    grad_ho = np.dot(Ohh,err_oh)
    grad_bho = np.sum(err_oh,axis=0)
    
    err_hh = np.dot(err_oh,np.transpose(Who))*delReLu(Zhh)
    grad_hh = np.dot(Oih,err_hh)
    grad_bhh = np.sum(err_hh,axis=0)
    
    err_hi = np.dot(err_hh,np.transpose(Whh))*delReLu(Zih)
    grad_ih = np.dot(X,err_hi)
    grad_bih = np.sum(err_hi,axis=0)
    
    """ Update Weights """
    Who -= alpha*grad_ho
    bho -= alpha*grad_bho
    Whh -= alpha*grad_hh
    bhh -= alpha*grad_bhh
    Wih -= alpha*grad_ih
    bih -= alpha*grad_bih

    
print("Cost Function after %d iterations is %f" % (iterations,j))    


np.array(J)
plt.plot(J)
plt.show()
