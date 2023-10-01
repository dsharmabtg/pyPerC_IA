# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:42:49 2020

@author: Dell
"""
import numpy as np
from Encoder import xlsread
from Encoder import centroidIT2
from Encoder import IA
from PlotFOU import plotIT2
from CWWEngine import LWA
from Decoder import Jaccard




if __name__ == "__main__":
    dataFilePath = "..\\data\\DATACOPY.xls"
    #Encoder of PerC
    A,words = xlsread(dataFilePath)
    row,col = A.shape
    MFs = np.zeros((int(col/2),9))
    nums = [None]*int(col/2)
    C = np.zeros((int(col/2),3))
    for i in range(int(col/2)):
         L = A[:,2*i]
         R = A[:,2*i+1]
         MFs[i,:],nums[i] = IA(L,R)
         C[i,:]=centroidIT2(MFs[i,:])

    #Plot the IT2
    plotIT2(words,MFs,"FOUDataPlot.png") 
    
    Xs=[0,1,2] 
    Ws=[3,4,5] 
    YLWA,_,_,_,_=LWA(MFs[Xs,:],MFs[Ws,:],21)   
    #print("YLWA",YLWA)
    tempYLWA = np.array(YLWA)
    tempYLWA = tempYLWA.reshape((1,9))
    plotIT2(["CWW Engine"],tempYLWA,"CWWPlot.png")
    
    S = np.zeros(int(col/2),dtype=float)
    for i in range(int(col/2)):
         S[i]=Jaccard(YLWA,MFs[i,:])

    index=np.argmax(S)
    maxS = np.max(S)
    decode=words[index]  #decoding
    print("MaxS:",maxS," Decode:",decode)