# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:42:22 2020

@author: Dell
"""

import numpy as np
from functools import reduce
import operator
from Encoder import EKM


def LWA(X,W,*args):
    
     varargin = args
     nargin = 2 + len(varargin)
     
     if nargin == 2:
          n = 21
     else:
          n = int(args[0])
     
     if X.shape[1] == 8:
          tempX = np.ones((X.shape[0],1),dtype=float)
          X = np.append(X,tempX,axis=1)
     
     if W.shape[1] == 8:
          tempW = np.ones((W.shape[0],1),dtype=float)
          W = np.append(W,tempW,axis=1)
     
         
     Yu,UMFYy,UMFYmu = FWA(X[:,0:4],W[:,0:4],n)
     
     Yl,LMFYy,LMFYmu = FWA(X[:,4:9],W[:,4:9],n)
     
     
     Y=reduce(operator.concat, [Yu[0:4],Yl])
     
     return Y,UMFYy,UMFYmu,LMFYy,LMFYmu

def FWA(X,W,*args):
     
     varargin = args
     nargin = 2 + len(varargin)
     
     
     
     #check whether the size of the vectors match
     if X.shape!=W.shape:
          raise Exception('Error: The sizes of the input vectors do not match. Abort.')
      
     if nargin == 2:
          n = 21
     else:
          n = int(args[0])
     
     N,M = X.shape
     
     if M == 4:
          tempX = np.ones((X.shape[0],1),dtype=float)
          X = np.append(X,tempX,axis=1)
     
     if W.shape[1] == 4:
          tempW = np.ones((W.shape[0],1),dtype=float)
          W = np.append(W,tempW,axis=1)
     

     
     hmin=np.min([X[:,4], W[:,4]])  #height of the FWA
     mu=hmin*np.hstack((np.linspace(0.0,1.0,n),np.linspace(1.0,0.0,n))) #mu-coordinates of the FWA

     a = np.zeros((N),dtype=float)
     b = np.zeros((N),dtype=float)
     c = np.zeros((N),dtype=float)
     d = np.zeros((N),dtype=float)
     y = np.zeros((2*n),dtype=float)
     
     for i in range(n):
         #a,b: alpha-cut on X
         for j in range(N):  #for each input, compute the alpha-cut
             # a,b: alpha-cut on X
             a[j]=X[j,0]+(X[j,1]-X[j,0])*mu[i]/float(X[j,4])
             
             b[j]=X[j,3]-(X[j,3]-X[j,2])*mu[i]/float(X[j,4])
             #c,d: alpha-cut on W
             c[j]=W[j,0]+(W[j,1]-W[j,0])*mu[i]/float(W[j,4])
             d[j]=W[j,3]-(W[j,3]-W[j,2])*mu[i]/float(W[j,4])         
         
         y[i]=EKM(a,c,d,-1)
         
         y[2*n-1-i]=EKM(b,c,d,1)
         
         
     
     #print("y:",y)
     #Y=[y([1 n n+1 2*n]) hmin]
     Y=[y[0], y[n-1], y[n], y[2*n-1],hmin]
     
     return Y,y,mu