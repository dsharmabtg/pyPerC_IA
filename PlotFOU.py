# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:44:41 2020

@author: Dell
"""


import numpy as np
import matplotlib.pyplot as plt
import random
from Encoder import mg 

def findIT2PlotPoint(**kwargs):
     #xUMF,uUMF,xLMF,uLMF,domain 
     #actualArg=['xUMF','uUMF','xLMF','uLMF','domain']
     tempArg=list()
     tempValue=list()
     for key, value in kwargs.items():
          tempArg.append(key)
          tempValue.append(np.array(value))
     
     for t in tempArg:
          if (t[0]!= 'xUMF' or t[1]!='uUMF' or t[2]!='uLMF' or t[3]!='domain') and len(tempArg)==4:
               raise Exception('The number of inputs must be 1, 2, 4 or 5.')
          if len(tempArg)>2:
               if len(t[0])!=len(t[1]):
                    raise Exception('xUMF and uUMF must have the same length.')
               if len(t[2])!=len(t[3]):
                    raise Exception('xLMF and uLMF must have the same length.')
               if len(tempArg)==4:
                    domain = [min(tempValue[0]), max(tempValue[0])]
          elif len(tempArg)==1:
               A = tempValue[0]               
               domain = np.array([A[0], A[3]])
               xUMF = np.linspace(domain[0],domain[1],100)
               xLMF = xUMF
               uUMF = mg(xUMF,A[0:4],np.array([0.0,1.0,1.0,0.0]))
               uLMF = mg(xLMF,A[4:8],np.array([0.0, A[8], A[8], 0.0]))
          elif len(tempArg)==2:
               A = tempValue[0]
               domain = tempValue[1]
               xUMF = np.linspace(domain[0],domain[1],100)
               xLMF = xUMF
               uUMF = mg(xUMF,A[0:4],np.array([0.0,1.0,1.0,0.0]))
               uLMF = mg(xLMF,A[4:8],np.array([0.0, A[8], A[8], 0.0]))     
               
     return xUMF, uUMF, xLMF, uLMF
     
            
def plotIT2(words,MF,fileName):
    
    fig = plt.figure(random.randint(1,100))
    DefaultSize = fig.get_size_inches()
    plt.rcParams["font.family"] = 'times new roman'
    plt.rcParams["font.weight"] = 'bold'
    plt.rcParams['axes.titlepad'] = 2
    plt.rcParams['axes.titlesize']=18
    #plt.rcParams['axes.labelsize']= 'x-large'
    plt.rcParams['text.latex.unicode']=True
    subPlotSize = findsubPlotGrid(len(words))
    for i in range(len(words)):
         ax=plt.subplot(subPlotSize[1],subPlotSize[0],i+1)
         xUMF, uUMF, xLMF, uLMF= findIT2PlotPoint(xUMF=MF[i,:])
         ax.fill(xUMF, uUMF, 'grey', xLMF, uLMF, 'w', alpha=1.0)
         ax.plot(xUMF, uUMF,'k', alpha=1.0)
         ax.plot(xLMF, uLMF,'k',alpha=1.0)
         ax.set(title=words[i])
         plt.xticks([])
         plt.yticks([])
         plt.axis([0, 10, 0, 1])
         plt.show()
         
    if fileName.find('.')==-1:
        fileName =  fileName + ".png"
        
    #plt.tight_layout(pad=0.05)
    fig.set_size_inches( (DefaultSize[0]*2, DefaultSize[1]*2) )
    fig.subplots_adjust(
              top=0.961,
              bottom=0.02,
              left=0.01,
              right=0.99,
              hspace=0.525,
              wspace=0.048
    )    
    plt.savefig(fileName)
     
         
     
def findsubPlotGrid(n):  
     tempFaclist = list()
     for i in range(1, int(pow(n, 1 / 2))+1): 
        if n % i == 0:
            tempFaclist.append([i,n/i])
                    
     tempFaclist = np.array(tempFaclist).astype(int)
     index = np.argmax(np.min(tempFaclist,axis=1))
     return tempFaclist[index]