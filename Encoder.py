# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:42:04 2020

@author: Dell
"""

import pandas as pd
import numpy as np
import math
import random 
from functools import reduce
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
  

def xlsread(path):
    
    dataframe = pd.read_excel(path)
    cn = dataframe.columns
    wordList = list()
    
    for i in range(0,len(cn),2):
        wordList.append(cn[i])
        
    
    Arr = np.asarray(dataframe)

    return Arr,wordList


def IA(L,R,*args):
       
    nums = dict()
    
    nums['OrgLength']=L.shape[0]
    varargin = args
    nargin = 2 + len(varargin)
    
    #Bad data processing, see Equation (1) in paper
    for i in range(L.shape[0]-1,-1,-1):
        if L[i]<0 or L[i]>10 or R[i]<0 or R[i]>10 or R[i]<L[i]:
            L = np.delete(L,i)
            R = np.delete(R,i)
    
    nums['BDLength']=L.shape[0]

    #Outlier processing, see Equation (2) in paper
    intLeng = R-L
    left = sorted(L)
    right = sorted(R)
    leng = sorted(intLeng)
    
    n = L.shape[0]

    NN1 = math.floor(n * 0.25 + 0.5)
    NN2 = math.floor(n * 0.75 + 0.5)
    
    #Compute Q(0.25), Q(0.75) and IQR for left-ends
    QL25 = (0.5 - n * 0.25 + NN1) * left[NN1-1] + (n * 0.25 + 0.5 - NN1) * left[NN1]
    QL75 = (0.5 - n * 0.75 + NN2) * left[NN2-1] + (n * 0.75 + 0.5 - NN2) * left[NN2]
    LIQR = QL75 - QL25

    
    
    #Compute Q(0.25), Q(0.75) and IQR for right-ends.
    QR25 = (0.5 - n * 0.25 + NN1) * right[NN1-1] + (n * 0.25 + 0.5 - NN1) * right[NN1]
    QR75 = (0.5 - n * 0.75 + NN2) * right[NN2-1] + (n * 0.75 + 0.5 - NN2) * right[NN2]
    RIQR = QR75 - QR25

    
    #Compute Q(0.25), Q(0.75) and IQR for interval length.
    QLeng25 = (0.5 - n * 0.25 + NN1) * leng[NN1-1] + (n * 0.25 + 0.5 - NN1) * leng[NN1]
    QLeng75 = (0.5 - n * 0.75 + NN2) * leng[NN2-1] + (n * 0.75 + 0.5 - NN2) * leng[NN2]
    lengIQR = QLeng75 - QLeng25
    bound=0.25
    
    
    #outlier processing
    for i in range(n-1,-1,-1):
        if (LIQR>bound and (L[i]<(QL25-1.5*LIQR) or L[i]>(QL75+1.5*LIQR))) or (RIQR>bound and (R[i]<(QR25-1.5*RIQR) or R[i]>(QR75+1.5*RIQR))) or (lengIQR>bound and (intLeng[i]<(QLeng25-1.5*lengIQR) or intLeng[i]>(QLeng75+1.5*lengIQR))):
            L = np.delete(L,i)
            R = np.delete(R,i)
            intLeng = np.delete(intLeng,i)
    

    nums['OLLength']=L.shape[0]        
    
    #Tolerance limit processing, see Equation (3) in paper
    n1 = L.shape[0]
    NN = 2000
    
    np.random.seed(2331)
    AA = np.floor(n1*np.random.rand(n1, NN)).astype(int)
    
    LA = L
    RA = R
    intLengA = intLeng
    
    resampleL = LA[AA]
    resampleR = RA[AA]
    resampleLeng = intLengA[AA]
 
    
    
    tempMeanL = np.mean(resampleL,axis=0)
    tempMeanR = np.mean(resampleR,axis=0)
    tempMeanLeng = np.mean(resampleLeng,axis=0)
   
    meanL = np.mean(tempMeanL)
    stdL = np.sqrt(n1)* np.std(tempMeanL,ddof=1)
    meanR = np.mean(tempMeanR) 
    stdR = np.sqrt(n1)* np.std(tempMeanR,ddof=1)
    meanLeng = np.mean(tempMeanLeng)
    stdLeng = np.sqrt(n1)* np.std(tempMeanLeng,ddof=1)
     
    
    K=np.array([32.019,32.019,8.380,5.369,4.275,3.712,3.369,3.136,2.967,2.839,2.737,2.655,2.587,2.529,2.48,2.437,2.4,2.366,2.337,2.31,2.31,2.31,2.31,2.31,2.208])
    
    k=K[min(L.shape[0]-1,24)]
    
    
    
    for i in range(L.shape[0]-1,-1,-1):
         if (stdL>bound and (L[i]<(meanL-k*stdL) or L[i]>(meanL + k*stdL))) or (stdR>bound and (R[i]<(meanR-k*stdR) or R[i]>(meanR + k*stdR))) or (stdLeng>bound and (intLeng[i]<(meanLeng-k*stdLeng) or intLeng[i]>(meanLeng + k*stdLeng))):
              L = np.delete(L,i)
              R = np.delete(R,i)
              intLeng = np.delete(intLeng,i)
              
    

    nums['TLLength']=L.shape[0]
    
    #Reasonable interval processing, see Equation (4)-(6) in paper
    n1 = L.shape[0]
    NN = 2000;
    
    np.random.seed(2231)
    AAR = np.floor(n1*np.random.rand(n1, NN)).astype(int)
    
    TLA = L
    TRA = R
    
    resampleRL = TLA[AAR]
    resampleRR = TRA[AAR]
    
    tempMeanRL = np.mean(resampleRL,axis=0)
    tempMeanRR = np.mean(resampleRR,axis=0)     
    
    meanRL = np.mean(tempMeanRL)
    stdRL = np.sqrt(n1)* np.std(tempMeanRL,ddof=1)
    meanRR = np.mean(tempMeanRR) 
    stdRR = np.sqrt(n1)* np.std(tempMeanRR,ddof=1)
     
     
    #Determine sigma*, see formula (5) in paper
    if stdRL+stdRR==0:
         barrier = (meanRL + meanRR)/2
    elif stdRL==0:
         barrier = meanRL+0.01
    elif stdRR==0:
         barrier = meanRR-0.01
    else:
         barrier1 =(-(meanRL*stdRR**2-meanRR*stdRL**2) + stdRL*stdRR*np.sqrt((meanRL-meanRR)**2+2*(stdRL**2-stdRR**2)*np.log(stdRL/stdRR)))/(stdRL**2-stdRR**2)
         barrier2 =(-(meanRL*stdRR**2-meanRR*stdRL**2) - stdRL*stdRR*np.sqrt((meanRL-meanRR)**2+2*(stdRL**2-stdRR**2)*np.log(stdRL/stdRR)))/(stdRL**2-stdRR**2)
         if  barrier1>=meanRL and barrier1<=meanRR:
              barrier = barrier1
         else:
              barrier = barrier2
     
   
    #Reasonable interval processing
    for i in range(L.shape[0]-1,-1,-1):
         if L[i] > barrier or R[i] < barrier:
              L = np.delete(L,i)
              R = np.delete(R,i)
              intLeng = np.delete(intLeng,i)
    
    
    nums['RIPLength']=L.shape[0]
    
    ml=np.mean(L)
    sl=np.std(L,ddof=1)
    mr=np.mean(R)
    sr=np.std(R,ddof=1)
    
    #Admissible region determination
    tTable=np.array([6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697, 1.684]) # alpha = 0.05;
    tAlpha=tTable[min(n-1,30)]
    meanL = np.mean(L)
    meanR = np.mean(R)
    
    newRIPLA = L
    newRIPRA = R
    
    C = newRIPRA - 5.831*newRIPLA
    D = newRIPRA - 0.171*newRIPLA - 8.29
    shift1 = tAlpha * np.std(C,ddof=1)/np.sqrt(n)
    shift2 = tAlpha * np.std(D,ddof=1)/np.sqrt(n)
    
    if np.isnan(shift1):
         shift1=0.0
    
    if np.isnan(shift2):
         shift2=0.0     
    
    FSL = np.zeros(L.shape[0]).astype(float)
    FSR = np.zeros(R.shape[0]).astype(float)
    #Establish nature of FOU, see Equation (19) in paper
    if (nargin==2 and meanR>(5.831*meanL-shift1)) or (nargin==3 and args[0]==1):
         for i in range(L.shape[0]-1,-1,-1):
              #left shoulder embedded T1 FS
              FSL[i] = 0.5*(L[i]+R[i]) - (R[i]-L[i])/np.sqrt(6)
              FSR[i] = 0.5*(L[i]+R[i]) + np.sqrt(6)*(R[i]-L[i])/3
              #Delete inadmissible T1 FSs
              if FSL[i]<0 or FSR[i]>10:
                   FSL= np.delete(FSL,i)
                   FSR= np.delete(FSR,i)                   
         # Compute the mathematical model for FOU(A~)
         UMF =[0.0,  0.0, float(np.max(FSL)), float(np.max(FSR))]
         LMF = [0.0, 0.0, float(np.min(FSL)), float(np.min(FSR)), 1.0]
    elif (nargin==2 and meanR>(8.29+0.171*meanL-shift2)) or (nargin==3 and args[0]==3):
          for i in range(L.shape[0]-1,-1,-1):
               # right shoulder embedded T1 FS
               FSL[i] = 0.5*(L[i]+R[i]) - (np.sqrt(6)*(R[i]-L[i]))/3
               FSR[i] = 0.5*(L[i]+R[i]) + (R[i]-L[i])/np.sqrt(6)
               #Delete inadmissible T1 FSs
               if FSL[i]<0 or FSR[i]>10:
                   FSL= np.delete(FSL,i)
                   FSR= np.delete(FSR,i)                    

          #Compute the mathematical model for FOU(A~)
          UMF =[float(np.min(FSL)), float(np.min(FSR)), 10.0, 10.0]
          LMF = [float(np.max(FSL)), float(np.max(FSR)), 10.0, 10.0, 1.0]
    else:
          for i in range(L.shape[0]-1,-1,-1):
               #internal embedded T1 FS
               FSL[i] = 0.5*(L[i]+R[i]) - np.sqrt(2)*0.5*(R[i]-L[i])
               FSR[i] = 0.5*(L[i]+R[i]) + np.sqrt(2)*0.5*(R[i]-L[i])
               #Delete inadmissible T1 FSs
               if FSL[i]<0 or FSR[i]>10:
                   FSL= np.delete(FSL,i)
                   FSR= np.delete(FSR,i)
        
          FSC=(FSL+FSR)/2
          #Compute the mathematical model for FOU(A~)
          L1 = float(np.min(FSL))
          L2 = float(np.max(FSL))
          R1 = float(np.min(FSR))
          R2 = float(np.max(FSR))
          C1 = float(np.min(FSC))
          C2 = float(np.max(FSC))
     
          temp = float(R1-C1)/float(C2-L2)
          apex = float(R1+temp*L2)/float(1+temp)
          height = float(R1-apex)/float(R1-C1)
          UMF =[L1, C1, C2, R2]
          LMF = [L2, apex, apex, R1, height]

    nums['FLength']=FSL.shape[0]
    nums['ml']=ml
    nums['sl']=sl
    nums['mr']=mr
    nums['sr']=sr
    
        
    MF = reduce(operator.concat, [UMF,LMF])
    
    return MF, nums

def EKM(xPoint,wLower,wUpper,maxFlag):
     
          
     if np.max(wUpper)==0.0 or np.max(xPoint)==0.0:
          y=0.0 
          return y
     
     if np.max(wLower)==0.0:
          if maxFlag>0:
               y=np.max(xPoint)
          else:
               y=np.min(xPoint)
     
          return y
     
     if xPoint.shape[0]==1:
          y=xPoint
          return y
     
        
     #combine zero firing intervals
     for iLoop in range(wUpper.shape[0]-1,-1,-1):
          if wUpper[iLoop]==0.0:
               xPoint = np.delete(xPoint,iLoop)
               wLower = np.delete(wLower,iLoop)
               wUpper = np.delete(wUpper,iLoop)
    
     
     xIndex = np.argsort(xPoint) 
     
     xSort  = np.array(sorted(xPoint))
    

     lowerSort = wLower[xIndex]
     upperSort = wUpper[xIndex]
         
     l = list()
     for p in range(xSort.shape[0]-1,-1,-1):
          if xSort[p]==0.0:
              l.append(p) 
     
     l = np.array(l)
     if l.shape[0] == 0:
          k = 0
     else:
          k = l[0]
     
     if k>0:
         xSort[0]=0
         lowerSort[0]=np.sum(lowerSort[0:k])
         upperSort[0]=np.sum(upperSort[0:k])
         for p in range(1,k):
              xSort = np.delete(xSort,p)
         
         for p in range(1,k):     
              lowerSort = np.delete(lowerSort,p)
         
         for p in range(1,k):     
              upperSort = np.delete(upperSort,p)
         
     

     ly=xSort.shape[0]

     if maxFlag < 0:
          k=round(ly/2.4)
          temp=np.hstack((upperSort[0:k],lowerSort[k:ly+1]))
     else:
          k=round(ly/1.7)
          temp=np.hstack((lowerSort[0:k],upperSort[k:ly+1]))
     
     ######################################33     
     a=np.dot(temp,xSort)
     b=np.sum(temp)
     y = float(a)/float(b)
     
     l = list()
     for p in range(xSort.shape[0]):
          if xSort[p]> y:
              l.append(p) 

     l = np.array(l)

     if l.shape[0] == 0:
          kNew=-1
     else:
          kNew = l[0]-1
     
     k = k-1
     
     while k!=kNew and kNew!=-1:
         mink=min(k,kNew)
         maxk=max(k,kNew)
         temp=upperSort[mink+1:maxk+1]-lowerSort[mink+1:maxk+1] #mink+1:maxk+2
         b=b-float(np.sign(kNew-k))*np.sign(maxFlag)*np.sum(temp)
         a=a-float(np.sign(kNew-k))*np.sign(maxFlag)*np.dot(temp,xSort[mink+1:maxk+1])
         y = float(a)/float(b)
         k=kNew
         l = list()
         for p in range(xSort.shape[0]):
              if xSort[p]> y:
                   l.append(p) 

         l = np.array(l)
         if l.shape[0] == 0:
             kNew = -1
              
         else:
             kNew = l[0]-1
     
     
     
     return y     

 
def mg(x,xMF,uMF):
     
     if len(xMF)!=len(uMF):
          raise Exception('xMF and uMF must have the same length.')
     
     index = np.argsort(xMF)
     xMF.sort()
     
     uMF = uMF[index]
     
     u = np.zeros(x.shape[0])
     
     for i in range(x.shape[0]):
          if x[i]<=xMF[0] or x[i]>=xMF[-1]:
               u[i]=0
          else:
               l = list()
               for j in range(xMF.shape[0]):
                    if xMF[j]<x[i]:
                         l.append(j)
               
               if len(l)==0:
                    left = 0
               else:
                    left = l[-1]
               
               right=left+1
               u[i]=uMF[left]+float(uMF[right]-uMF[left])*(x[i]-xMF[left])/float(xMF[right]-xMF[left])
    
     return u
     
     
def centroidIT2(MF):

     if MF.shape[0]!= 9:
          raise Exception('The input vector must be a 9-point representation of an IT2 FS.')
          

     Xs=np.linspace(MF[0],MF[3],100)
     
     UMF=mg(Xs,MF[0:4],np.array([0.0,1.0,1.0,0.0]))
     
     LMF=mg(Xs,MF[4:8],np.array([0.0,float(MF[8]),float(MF[8]),0.0]))
     
     Cl=EKM(Xs,LMF,UMF,-1)
     Cr=EKM(Xs,LMF,UMF,1)
     Cavg=float(Cl+Cr)/2.0

     return Cl,Cr,Cavg
