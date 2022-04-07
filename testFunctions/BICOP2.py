# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:56:36 2020

@author: r.dewinter
"""


import numpy as np

def BICOP2(x):
    a = 0.1
    b = 0.9
    
    gx = np.sum(x[1:] - np.sin(0.5*np.pi*x[0]))**2
    
    f1 = x[0] + gx
    f2 = 1 - x[0]**2 + gx
    
    g1 = gx - a
    g2 = b - gx
    
    c1 = g1
    c2 = g2
    #-1* constr because of sacobra's constraint handling
    return [ np.array([f1, f2]), -1*np.array([c1,c2]) ]


# amount = 1000000
# x = np.random.rand(amount*10)
# x = np.reshape(x, (amount, 10))
# objs = np.zeros((amount,2))
# cons = np.zeros((amount,2))
# for i in range(len(x)):
#     objs[i], cons[i] = BICOP2(x[i])
    
# import matplotlib.pyplot as plt
# plt.plot(objs[:,0], objs[:,1], 'ro')
# plt.plot(objs[np.sum(cons<=0,axis=1)==2][:,0], objs[np.sum(cons<=0,axis=1)==2][:,1], 'bo')