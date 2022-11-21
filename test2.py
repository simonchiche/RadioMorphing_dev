#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:35:43 2022

@author: chiche
"""


import numpy as np

a = np.ones([10,10])

'''
for i in range(len(a)):
    print(a[i])
    if(a[i]== 2): np.append(a,20)
 '''   

IndexAll = []

for i in range(5):
    
    IndexAll.append(np.random.randint(5, size=4))
    
#IndexAll[1]  =np.append(IndexAll[1],20)


for i in range(len(IndexAll)):
    
   for j in range(len(IndexAll[i])):
       
       print(i, j, IndexAll[i][j])
       if(i ==1): IndexAll[i+1]  =np.append(IndexAll[i+1],20)