# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:32:48 2021

@author: r.dewinter
"""
import numpy as np
from multiprocessing import freeze_support

from pSAMO_COBRA_Init import pSAMO_COBRA_Init
from pSAMO_COBRA_PhaseII import pSAMO_COBRA_PhaseII

# example test function with two variables, two constraints, two objectives
def BNH(x):
    f1 = 4*x[0]**2+4*x[1]**2
    f2 = (x[0]-5)**2 + (x[1]-5)**2
    
    c1 = -1*((x[0]-5)**2 + x[1]-25)
    c2 = (x[0]-8)**2 + (x[1]-3)**2 - 7.7
    #-1* because constraints should be smaller then 0.
    return [ np.array([f1, f2]), -1*np.array([c1,c2]) ]



if __name__ == '__main__':  
    freeze_support() # you need this when you want to use the multiprocessing part of the optimization algorithm.

    fn = BNH # reference to test function which takes as input x between lower and upper bound.
    lower = np.array([0,0]) # lower limit of decision variables
    upper = np.array([5,3]) #upper limit of decision variables
    ref = np.array([140,50]) #Reference point:  worst possible objective score per objective 
    nConstraints = 2 # number of constraints
    batch = 5 # number of solutions proposed per iteration, set to 1 for one solution at a time. 
    # One solution per iteration usually leads to less required function evaluations to find the pareto frontier, 
    # however if you can evaluate objective function in parrallel, you can chose a higher number and this way save wall clock time
    feval = 80 # number of allowed functionevaluations, so in case of 5 solutions per iteration, you will do 80/5=16 iterations
    useAllCores = True # define if you want to use all cores in during the process, if false you will only use one core
    cobra = pSAMO_COBRA_Init(fn, nConstraints, ref, lower, upper, feval, batch=batch, useAllCores=useAllCores) # initilization phase
    cobra = pSAMO_COBRA_PhaseII(cobra) # optimization phase
    # in cobra dictionary you can find a lot of details about the optimization run
    
    