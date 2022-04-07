# Multi-Point-SAMO-COBRA
The SAMO-COBRA algorithm extended with a multi point acquisition function and parallelism so that objectives and constraints can be run in parallel.

Bayesian optimization is often used to optimize expensive black box optimization problems with long simulation times. Typically Bayesian optimization algorithms propose one solution per iteration. The downside of this strategy is the sub-optimal use of available computing power. To efficiently use the available computing power (or a number of licenses etc.) we introduce a multi-point acquisition function for parallel efficient multi-objective optimization algorithms. The multi-point acquisition function is based on the hypervolume contribution of multiple solutions simultaneously, leading to well spread solutions along the Pareto frontier. By combining this acquisition function with a constraint handling technique, multiple feasible solutions can be proposed and evaluated in parallel every iteration. The hypervolume and feasibility of the solutions can easily be estimated by using multiple cheap radial basis functions as surrogates with different configurations. The acquisition function can be used with different population sizes and even for one shot optimization. The strength and generalizability of the new acquisition function is demonstrated by optimizing a set of black box constraint multi-objective problem instances. The experiments show a huge time saving factor by using our novel multi-point acquisition function, while only marginally worsening the hypervolume after the same number of function evaluations. 

```python
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
```
