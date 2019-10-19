# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:12:34 2019

HW3 Neural Networks from Scartch

@author: GargyaGokhale

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


""" Necessary Function Definitions """
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
    Z = ((1-Y)/(1-O + np.exp(-15))) - (Y/(O + np.exp(-15)))
    return Z

""" Class for Neural Network """

class NerualNetwork:
    
    """ Defining weights and bias """
    
    np.random.seed(1)
    a=0.05

    Wih = a*np.random.rand(2,10)
    bih = np.random.rand(1,10)
    
    Whh = a*np.random.rand(10,10)
    bhh = np.random.rand(1,10)

    Who = a*np.random.rand(10,1)
    bho = np.random.rand(1,1)
    
    """ Feed Forward """
    def FeedForward(self,X):
        self.Zih = np.dot(np.transpose(X),self.Wih) + self.bih
        self.Oih = np.transpose(reLu(self.Zih))
        
        self.Zhh = np.dot(np.transpose(self.Oih),self.Whh) + self.bhh
        self.Ohh = np.transpose(reLu(self.Zhh))
        
        self.Zho = np.dot(np.transpose(self.Ohh),self.Who) + self.bho
        self.O = sigmoid(self.Zho)
        return 
    
    def BackPropogate(self,X,Y,alpha,n):
        """ alpha is the learning rate
            n is the batch size """
            
        err_oh = delCrossEntropy(self.O,Y)*delSigmoid(self.Zho)/n
        grad_ho = np.dot(self.Ohh,err_oh)
        grad_bho = np.sum(err_oh,axis=0)
        
        err_hh = np.dot(err_oh,np.transpose(self.Who))*delReLu(self.Zhh)
        grad_hh = np.dot(self.Oih,err_hh)
        grad_bhh = np.sum(err_hh,axis=0)
        
        err_hi = np.dot(err_hh,np.transpose(self.Whh))*delReLu(self.Zih)
        grad_ih = np.dot(X,err_hi)
        grad_bih = np.sum(err_hi,axis=0)
        
        """ Update Weights """
        self.Who -= alpha*grad_ho
        self.bho -= alpha*grad_bho
        self.Whh -= alpha*grad_hh
        self.bhh -= alpha*grad_bhh
        self.Wih -= alpha*grad_ih
        self.bih -= alpha*grad_bih
        return

""" Main Script """

print("Begin")
data = pd.read_excel("HW3Atrain.xlsx")
TotalData = len(data)
batches = 10
n = TotalData//batches

np.random.seed(1)

A = NerualNetwork()

b1=0
b2=1
J=[]
alpha = 0.0001
count=[]
for batch in range(batches):
    b1 = batch
    b2 = batch + 1
    X0 = np.reshape(np.array(data.X_0[b1*n:b2*n]),(1,n))
    X1 = np.reshape(np.array(data.X_1[b1*n:b2*n]),(1,n))
    Y = np.reshape(np.array(data.y[b1*n:b2*n]),(n,1))
    X = np.append(X0,X1,axis=0)


    iterations = 10000
    for itr in range(iterations):
        A.FeedForward(X)
        j = np.mean(crossEntropy(A.O,Y))
        J.append(j)
        count.append(batch*iterations + itr + 1)
        A.BackPropogate(X,Y,alpha,n)

    print("Cost Function after batch %d \n %d iterations is %f" % (batch+1,iterations,j))


np.array(J)
plt.plot(J)
plt.show()


""" Validation """

data_valid = pd.read_excel("HW3Avalidate.xlsx")
X0_val = np.reshape(np.array(data_valid.X_0),(1,len(data_valid)))
X1_val = np.reshape(np.array(data_valid.X_1),(1,len(data_valid)))
Y_val = np.reshape(np.array(data_valid.y),(len(data_valid),1))
X_val = np.append(X0_val,X1_val,axis=0)



A.FeedForward(X_val)

fin_err = A.O - Y_val
mat = np.zeros((len(Y_val),1))
cr = []      #Correct prediction
wr = []     #wrong prediction
unc = []     #uncertain


for i in range(len(Y_val)):
    if abs(fin_err[i]) <= 0.45:
        cr.append(i)
    elif abs(fin_err[i]) >0.45 and abs(fin_err[i])<0.55 :
        unc.append(i)
    else:
        wr.append(i)

print("Prediction Results")
print("Number of correct predictions is %d" % len(cr))
print("Number of wrong predictions is %d" % len(wr))
print("Number of uncertain predictions is %d" % len(unc))




