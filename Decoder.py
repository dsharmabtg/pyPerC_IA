# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:42:34 2020

@author: Dell
"""


import numpy as np
from Encoder import mg

def Jaccard(A,B):
     
     A = np.array(A)
     B = np.array(B)
     N=200 #number of discretizations
     minX=min(A[0],B[0]) # the range
     maxX=max(A[3],B[3])
     X=np.linspace(minX,maxX,N)

     lowerA=mg(X,A[4:8],np.array([0.0, float(A[8]), float(A[8]), 0.0]))
     upperA=mg(X,A[0:4],np.array([0.0,1.0,1.0,0.0]))
     lowerB=mg(X,B[4:8],np.array([0.0, float(B[8]), float(B[8]), 0.0]))
     upperB=mg(X,B[0:4],np.array([0.0,1.0,1.0,0.0]))

     
     S=np.sum(np.hstack((np.minimum(upperA,upperB),np.minimum(lowerA,lowerB))))/np.sum(np.hstack((np.maximum(upperA,upperB),np.maximum(lowerA,lowerB))))

     return S