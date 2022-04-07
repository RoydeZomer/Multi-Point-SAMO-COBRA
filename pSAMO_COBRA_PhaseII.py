# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:48:42 2017

@author: r.dewinter
"""

from SACOBRA import plog
from SACOBRA import plogReverse
from SACOBRA import standardize_obj
from SACOBRA import rescale_constr

from RbfInter import trainRBF
from RbfInter import interpRBF
from RbfInter import distLine

from lhs import lhs 
from hypervolume import hypervolume
from paretofrontFeasible import paretofrontFeasible
from visualiseParetoFront import visualiseParetoFront

import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import time
import warnings
import copy
import os
from functools import partial


def getConstraintPrediction(x, surrogateModels, bestPredictor, nConstraints, GresPlogRescaledDivider, GresRescaledDivider, EPS=None):
    constraintPredictions = np.zeros(nConstraints)
    for coni in range(nConstraints):
        conKernel = bestPredictor['conKernel'][coni]
        conLogStr = bestPredictor['conLogStr'][coni]
        surrogate = surrogateModels[conKernel]['Constraints'][conLogStr][coni]
        
        if conLogStr=='PLOGrescaled':
            constraintPrediction = interpRBF(np.array(x), surrogate)
            constraintPrediction = GresPlogRescaledDivider[coni] * constraintPrediction
            constraintPrediction = plogReverse(constraintPrediction)
        else:
            constraintPrediction = interpRBF(np.array(x), surrogate)
            constraintPrediction = GresRescaledDivider[coni] * constraintPrediction                
        
        if EPS is None:
            constraintPredictions[coni] = constraintPrediction
        else:
            constraintPredictions[coni] = constraintPrediction+EPS[coni]**2

    return constraintPredictions

def gCOBRA(x, A, lower, upper, surrogateModels, bestPredictor, nConstraints, GresPlogRescaledDivider, GresRescaledDivider, EPS):
    h = 0
    distance = distLine(x, A)
    if any(distance<=(len(A[0])/1e4)):
        if min(distance) != 0:
            h = min(distance)**-2
        else:
            h = np.finfo(np.float64).max
    if not all(np.isfinite(distance)):
        h = np.finfo(np.float64).max
        
    constraintPrediction = getConstraintPrediction(x, surrogateModels, bestPredictor, nConstraints, GresPlogRescaledDivider, GresRescaledDivider, EPS)
    
    if np.any(np.isnan(constraintPrediction)):
        warnings.warn('gCOBRA: constraintPrediction value is NaN, returning Inf',DeprecationWarning)
        return([np.finfo(np.float64).min]*(len(lower)*2+len(constraintPrediction)+1))
    
    boundaries = []
    for i in range(len(lower)):
        boundaries.append(lower[i] - x[i])
    for i in range(len(upper)):
        boundaries.append(x[i] - upper[i])
        
    h = np.append(np.array([-1*h]), -1*constraintPrediction) #cobyla treats positive values as feasible
    h = np.append(h, -1*np.array(boundaries))
    return(h)

def batch_gCOBRA(x, batch, A, lower, upper, surrogateModels, bestPredictor, nConstraints, GresPlogRescaledDivider, GresRescaledDivider, EPS):
    h = np.array([])
    size = int(len(x)/batch)
    for i in range(batch):
        xi = x[i*size:(i+1)*size]
        h = np.append(h, gCOBRA(xi, A, lower, upper, surrogateModels, bestPredictor, nConstraints, GresPlogRescaledDivider, GresRescaledDivider, EPS))
    return(h)    

def get_potentialSolution(x, surrogateModels, bestPredictor, nObj, infillCriteria, FresPlogStandardizedStd, FresStandardizedStd, FresPlogStandardizedMean, FresStandardizedMean):
    potentialSolution = np.zeros(nObj)
    for obji in range(nObj):
        objKernel = bestPredictor['objKernel'][obji]
        objLogStr = bestPredictor['objLogStr'][obji]
        surrogate = surrogateModels[objKernel]['Objectives'][objLogStr][obji]
        
        potsol = 0
        uncertainty = 0
        if objLogStr=='PLOGStandardized':
            if infillCriteria == 'PHV':
                potsol = interpRBF(x, surrogate, uncertainty=False)                
                potsol = potsol*FresPlogStandardizedStd[obji] + FresPlogStandardizedMean[obji]
                potsol = plogReverse(potsol)
            elif infillCriteria == 'SMS':
                potsol, uncertainty = interpRBF(x, surrogate, uncertainty=True)
                uncertainty = uncertainty * FresStandardizedStd[obji]
                potsol = potsol*FresPlogStandardizedStd[obji] + FresPlogStandardizedMean[obji]
                potsol = plogReverse(potsol)
            else:
                raise ValueError("This infill criteria is not implemented")
        else:
            if infillCriteria == 'PHV':
                potsol = interpRBF(x, surrogate, uncertainty=False)
                potsol = potsol*FresStandardizedStd[obji] + FresStandardizedMean[obji]
            elif infillCriteria == 'SMS':
                potsol, uncertainty = interpRBF(x, surrogate, uncertainty=True)
                potsol = potsol*FresStandardizedStd[obji] + FresStandardizedMean[obji]
                uncertainty = uncertainty * FresStandardizedStd[obji]
            else:
                raise ValueError("This infill criteria is not implemented")
        
        potentialSolution[obji] = potsol - np.abs(uncertainty)
    return potentialSolution
    
def compute_infill_criteria_score(x, surrogateModels, bestPredictor, nObj, infillCriteria, FresPlogStandardizedStd, FresStandardizedStd, FresPlogStandardizedMean, FresStandardizedMean, currentHV, paretoFrontier, ref):
    if np.any(np.isnan(x)):
        return np.finfo(np.float64).max
    if not all(np.isfinite(x)):
        return np.finfo(np.float64).max

    potentialSolution = get_potentialSolution(x, surrogateModels, bestPredictor, nObj, infillCriteria, FresPlogStandardizedStd, FresStandardizedStd, FresPlogStandardizedMean, FresStandardizedMean)
    if not all(np.isfinite(potentialSolution)):
        return np.finfo(np.float64).max
    
    penalty = 0
    ##### add epsilon?
    logicBool = np.all(paretoFrontier<= potentialSolution, axis=1)
    for j in range(paretoFrontier.shape[0]):
        if logicBool[j]:
            p = - 1 + np.prod(1 + (potentialSolution-paretoFrontier[j,:]))
            penalty = max(penalty, p)
    if penalty == 0: #non-dominated solutions
        potentialFrontier = np.append(paretoFrontier, [potentialSolution], axis=0)
        myhv = hypervolume(potentialFrontier, ref)
        f = currentHV - myhv
    else:
        f = penalty
    return f

def batch_infill_criteria_score(x, batch, surrogateModels, bestPredictor, nObj, infillCriteria, FresPlogStandardizedStd, FresStandardizedStd, FresPlogStandardizedMean, FresStandardizedMean, currentHV, paretoFrontier, ref):
    if np.any(np.isnan(x)):
        return np.finfo(np.float64).max
    if not all(np.isfinite(x)):
        return np.finfo(np.float64).max
    
    solutions = []
    penalties = []
    
    size = int(len(x)/batch)
    for i in range(batch):
        xi = x[i*size:(i+1)*size]
    
        potentialSolution = get_potentialSolution(xi, surrogateModels, bestPredictor, nObj, infillCriteria, FresPlogStandardizedStd, FresStandardizedStd, FresPlogStandardizedMean, FresStandardizedMean)
        solutions.append(potentialSolution)
        if not all(np.isfinite(potentialSolution)):
            return np.finfo(np.float64).max
        
        penalty = 0
        ##### add epsilon?
        logicBool = np.all(paretoFrontier<= potentialSolution, axis=1)
        for j in range(paretoFrontier.shape[0]):
            if logicBool[j]:
                p = - 1 + np.prod(1 + (potentialSolution-paretoFrontier[j,:]))
                penalty = max(penalty, p)
        penalties.append(penalty)
    potentialFrontier = np.append(paretoFrontier, solutions, axis=0)
    myhv = hypervolume(potentialFrontier, ref)
    f = currentHV - myhv + sum(penalties)
    return f

def pool_job(xStart, criteria_function=None, cons=None, seqFeval=None, seqTol=None):
    opts = {'maxiter':seqFeval, 'tol':seqTol}
    subMin = optimize.minimize(criteria_function, xStart, constraints=cons, options=opts, method='COBYLA')
    return subMin


def pSAMO_COBRA_PhaseII(cobra):
    # print("PHASE II started")
    phase = 'PHASE II'
    pool = cobra['pool']

    if cobra['hypervolumeProgress'] is None:
        raise ValueError("cobraPhaseII: cobra['hypervolumeProgress'] is None! First run smscobraInit")
        
    fn = cobra['fn']
    n = len(cobra['A'])
    if n==cobra['initDesPoints']:
        predHV = np.empty(cobra['initDesPoints'])
        predHV[:] = np.nan # structure to store surrogate optimization results
        cobra['optimizerConvergence'] = np.ones(cobra['initDesPoints']) # vector to store optimizer convergence
        feval = np.empty(cobra['initDesPoints'])
        feval[:] = np.nan
        
    if n >= cobra['feval']:
        raise ValueError("ERROR! Number of function evaluations after initialization is larger than total allowed evaluations")
    
    def define_best_predictor(x, yTrue, conTrue, surrogateModels, cobra): 
        if x is not None: # check if dict is empty in first iteration.
            for obji in range(cobra['nObj']):
                for kernel in cobra['RBFmodel']:
                    surrogatePlog = surrogateModels[kernel]['Objectives']['PLOGStandardized'][obji]
                    plogSol = interpRBF(x,surrogatePlog)
                    plogSol = plogSol*cobra['FresPlogStandardizedStd'][obji] + cobra['FresPlogStandardizedMean'][obji]
                    plogSol = plogReverse(plogSol)
                    cobra['SurrogateErrors']['OBJ'+str(obji)+'PLOG'+kernel].append((plogSol - yTrue[obji])**2)
                    
                    surrogate = surrogateModels[kernel]['Objectives']['Standardized'][obji]
                    sol = interpRBF(x, surrogate)
                    sol = sol*cobra['FresStandardizedStd'][obji] + cobra['FresStandardizedMean'][obji]
                    cobra['SurrogateErrors']['OBJ'+str(obji)+kernel].append((sol - yTrue[obji])**2)
            
            for coni in range(cobra['nConstraints']):
                for kernel in cobra['RBFmodel']:
                    surrogatePlog = surrogateModels[kernel]['Constraints']['PLOGrescaled'][coni]
                    plogSol = interpRBF(x, surrogatePlog)
                    plogSol = cobra['GresPlogRescaledDivider'][coni] * plogSol
                    plogSol = plogReverse(plogSol)                    
                    cobra['SurrogateErrors']['CON'+str(coni)+'PLOG'+kernel].append((plogSol - conTrue[coni])**2)
                    
                    surrogate = surrogateModels[kernel]['Constraints']['Rescaled'][coni]
                    sol = interpRBF(x, surrogate)
                    sol = cobra['GresRescaledDivider'][coni] * sol
                    cobra['SurrogateErrors']['CON'+str(coni)+kernel].append((sol - conTrue[coni])**2)
        
        tempErrors = copy.deepcopy(cobra['SurrogateErrors'])
                
        hvgrowing = np.zeros(len(cobra['hypervolumeProgress']))==1
        for i in range(1,len(cobra['hypervolumeProgress'])):    
            if cobra['hypervolumeProgress'][i] - cobra['hypervolumeProgress'][i-1] > 0:
                hvgrowing[i] = True
        
        hvgrowing[-2*cobra['batch']:] = True # besides the hypervolume improvement iterations, the results from the last two iterations are also taken into account
        
        bestPredictor = {'objKernel':[], 'objLogStr':[], 'conKernel':[], 'conLogStr':[]}
        for obji in range(cobra['nObj']):
            minScore = np.finfo(np.float64).max
            bestPredictor['objKernel'].append(cobra['RBFmodel'][-1])
            bestPredictor['objLogStr'].append('Standardized')
            for kernel in cobra['RBFmodel']:
                if np.sqrt(np.mean(np.array(tempErrors['OBJ'+str(obji)+kernel])[hvgrowing])) < minScore:
                    minScore = np.sqrt(np.mean(np.array(tempErrors['OBJ'+str(obji)+kernel])[hvgrowing]))
                    bestPredictor['objKernel'][obji] = kernel
                    bestPredictor['objLogStr'][obji] = 'Standardized'
                if np.sqrt(np.mean(np.array(tempErrors['OBJ'+str(obji)+'PLOG'+kernel])[hvgrowing])) < minScore:
                    minScore = np.sqrt(np.mean(np.array(tempErrors['OBJ'+str(obji)+'PLOG'+kernel])[hvgrowing]))
                    bestPredictor['objKernel'][obji] = kernel
                    bestPredictor['objLogStr'][obji] = 'PLOGStandardized'
        for coni in range(cobra['nConstraints']):
            minScore = np.finfo(np.float64).max
            bestPredictor['conKernel'].append(cobra['RBFmodel'][-1])
            bestPredictor['conLogStr'].append('Rescaled')
            for kernel in cobra['RBFmodel']:
                if np.sqrt(np.mean(np.array(tempErrors['CON'+str(coni)+kernel])[hvgrowing])) < minScore:
                    minScore = np.sqrt(np.mean(np.array(tempErrors['CON'+str(coni)+kernel])[hvgrowing]))
                    bestPredictor['conKernel'][coni] = kernel
                    bestPredictor['conLogStr'][coni] = 'Rescaled'
                if np.sqrt(np.mean(np.array(tempErrors['CON'+str(coni)+'PLOG'+kernel])[hvgrowing])) < minScore:
                    minScore = np.sqrt(np.mean(np.array(tempErrors['CON'+str(coni)+'PLOG'+kernel])[hvgrowing]))
                    bestPredictor['conKernel'][coni] = kernel
                    bestPredictor['conLogStr'][coni] = 'PLOGrescaled'    
        
        cobra['bestPredictor'].append(bestPredictor)     
        
    def updateInfoAndCounters(cobra, xNew, yNewEval, conNewEval, phase, surrogateModels):
        cobra['Fsteepness'] = [0] * cobra['nObj']
        cobra['A'] = np.vstack((cobra['A'], xNew))
        cobra['lastX'] = xNew
        cobra['Fres'] = np.vstack((cobra['Fres'], yNewEval))
        cobra['Gres'] = np.vstack((cobra['Gres'], conNewEval))
        
        FresStandardized = np.full_like(cobra['Fres'], 0)
        FresStandardizedMean = np.zeros(cobra['nObj'])
        FresStandardizedStd = np.zeros(cobra['nObj'])

        FresPlogStandardized = np.full_like(cobra['Fres'], 0)
        FresPlogStandardizedMean = np.zeros(cobra['nObj'])
        FresPlogStandardizedStd = np.zeros(cobra['nObj'])
        
        for obji in range(cobra['nObj']):
            res, mean, std = standardize_obj(cobra['Fres'][:,obji])        
            FresStandardized[:,obji] = res
            FresStandardizedMean[obji] = mean
            FresStandardizedStd[obji] = std
            
            plogFres = plog(cobra['Fres'][:,obji])
            res, mean, std = standardize_obj(plogFres)
            FresPlogStandardized[:,obji] = res
            FresPlogStandardizedMean[obji] = mean 
            FresPlogStandardizedStd[obji] = std
            
        cobra['FresStandardized'] = FresStandardized
        cobra['FresStandardizedMean'] = FresStandardizedMean
        cobra['FresStandardizedStd'] = FresStandardizedStd  
        cobra['lastF'] = FresStandardized[-1]
        
        cobra['FresPlogStandardized'] = FresPlogStandardized
        cobra['FresPlogStandardizedMean'] = FresPlogStandardizedMean
        cobra['FresPlogStandardizedStd'] = FresPlogStandardizedStd        

        GresRescaled = np.full_like(cobra['Gres'], 0)
        GresRescaledDivider = np.zeros(cobra['nConstraints'])
        GresPlogRescaled = np.full_like(cobra['Gres'], 0)
        GresPlogRescaledDivider = np.zeros(cobra['nConstraints'])
        for coni in range(cobra['nConstraints'] ):
            GresRescaled[:,coni], GresRescaledDivider[coni] = rescale_constr(cobra['Gres'][:,coni])
            plogGres = plog(cobra['Gres'][:,coni])
            GresPlogRescaled[:,coni], GresPlogRescaledDivider[coni] = rescale_constr(plogGres)
        
        cobra['GresRescaled'] = GresRescaled
        cobra['GresRescaledDivider'] = GresRescaledDivider
        
        cobra['GresPlogRescaled'] = GresPlogRescaled
        cobra['GresPlogRescaledDivider'] = GresPlogRescaledDivider
        
        
        pff = paretofrontFeasible(cobra['Fres'], cobra['Gres'])
        pf = cobra['Fres'][pff]
        cobra['paretoFrontier'] = pf
        cobra['paretoFrontierFeasible'] = pff
        
        hv = hypervolume(pf, cobra['ref'])
        cobra['currentHV'] = hv
        
        newNumViol = np.sum(conNewEval > 0)
        newMaxViol = max(0, max(conNewEval))
        
        if newNumViol == 0:
            cobra['hypervolumeProgress'] = np.append(cobra['hypervolumeProgress'], hv)
        else:
            cobra['hypervolumeProgress'] = np.append(cobra['hypervolumeProgress'], cobra['hypervolumeProgress'][-1])
        
        cobra['numViol'] = np.append(cobra['numViol'], newNumViol)
        cobra['maxViol'] = np.append(cobra['maxViol'], newMaxViol)
        cobra['phase'].append(phase)
        
        for ci in range(cobra['nConstraints']):
            if conNewEval[ci] <= 0:
                cobra['EPS'][ci] = cobra['EPS'][ci]*(1-cobra['epsilonLearningRate'])
            else:
                cobra['EPS'][ci] = np.minimum((cobra['epsilonLearningRate']+1)*cobra['EPS'][ci],cobra['epsilonMax'][ci])
                
        define_best_predictor(xNew, yNewEval, conNewEval, surrogateModels, cobra)
        return(cobra)

    def trainSurrogates(cobra):
        surrogateModels = {}
        A = cobra['A']
        FresStandardized = cobra['FresStandardized'].T
        FresPlogStandardized = cobra['FresPlogStandardized'].T
        GresRescaled = cobra['GresRescaled'].T
        GresPlogRescaled = cobra['GresPlogRescaled'].T
        kernels = cobra['RBFmodel']
        
        for kernel in kernels:
            surrogateModels[kernel] = {'Constraints':{}, 'Objectives':{}}
            surrogateModels[kernel]['Constraints'] = {'PLOGrescaled':[], 'Rescaled':[]}
            surrogateModels[kernel]['Objectives'] = {'PLOGStandardized':[], 'Standardized':[]}
            for g in GresRescaled:
                surrogateModels[kernel]['Constraints']['Rescaled'].append(trainRBF(A,g,ptail=True,squares=True,smooth=0.00,rbftype=kernel))

            for g in GresPlogRescaled:
                surrogateModels[kernel]['Constraints']['PLOGrescaled'].append(trainRBF(A,g,ptail=True,squares=True,smooth=0.00,rbftype=kernel))

            for f in FresStandardized:
                surrogateModels[kernel]['Objectives']['Standardized'].append(trainRBF(A,f,ptail=True,squares=True,smooth=0.00,rbftype=kernel))

            for f in FresPlogStandardized:
                surrogateModels[kernel]['Objectives']['PLOGStandardized'].append(trainRBF(A,f,ptail=True,squares=True,smooth=0.00,rbftype=kernel))
    
        return surrogateModels

    def computeStartPoints(cobra, points):
        np.random.seed(cobra['cobraSeed']+len(cobra['A'])+1)
        lb = cobra['lower']
        ub = cobra['upper']
        strategy = cobra['computeStartPointsStrategy']
        
        if strategy=='random':
            startPoints = lb + np.random.rand(len(lb)) * (ub-lb)
        
        elif strategy=='multirandom':
            startPoints = np.random.rand(len(lb)*points)
            startPoints = startPoints.reshape((points,len(lb)))
            startPoints = lb + startPoints * (ub-lb)
        
        elif strategy=='LHS':
            startPoints = lhs(len(lb), samples=points, criterion="center", iterations=5)
            
        elif strategy=='midle':
            startPoints = (lb + ub)/ 2
        else:
            # do something smart?
            raise ValueError("This strategy does not exist for computeStartPoints")
            
        return startPoints
    
    def findSurrogateMinimum(cobra, surrogateModels, bestPredictor, pool):
        submins = []
        besti = 0
        bestFun = 0
        success = []
        cons = []
        
        if cobra['batch'] == 1:
            xStarts = computeStartPoints(cobra, cobra['computeStartingPoints'])
            gCOBRA_partial = partial(gCOBRA, A=cobra['A'], lower=cobra['lower'], upper=cobra['upper'], surrogateModels=surrogateModels, bestPredictor=bestPredictor, nConstraints=cobra['nConstraints'], GresPlogRescaledDivider=cobra['GresPlogRescaledDivider'], GresRescaledDivider=cobra['GresRescaledDivider'], EPS=cobra['EPS'])
            cons.append({'type':'ineq','fun':gCOBRA_partial})
            compute_infill_criteria_score_partial = partial(compute_infill_criteria_score, surrogateModels=surrogateModels, bestPredictor=bestPredictor, nObj=cobra['nObj'], infillCriteria=cobra['infillCriteria'], FresPlogStandardizedStd=cobra['FresPlogStandardizedStd'], FresStandardizedStd=cobra['FresStandardizedStd'], FresPlogStandardizedMean=cobra['FresPlogStandardizedMean'], FresStandardizedMean=cobra['FresStandardizedMean'], currentHV=cobra['currentHV'], paretoFrontier=cobra['paretoFrontier'], ref=cobra['ref'])
        else:
            xStarts = computeStartPoints(cobra, cobra['computeStartingPoints']*cobra['batch'])
            xStarts = xStarts.reshape((cobra['computeStartingPoints'],len(cobra['lower'])*cobra['batch']))
            gCOBRA_partial = partial(batch_gCOBRA, batch=cobra['batch'], A=cobra['A'], lower=cobra['lower'], upper=cobra['upper'], surrogateModels=surrogateModels, bestPredictor=bestPredictor, nConstraints=cobra['nConstraints'], GresPlogRescaledDivider=cobra['GresPlogRescaledDivider'], GresRescaledDivider=cobra['GresRescaledDivider'], EPS=cobra['EPS'])
            cons.append({'type':'ineq','fun':gCOBRA_partial})
            compute_infill_criteria_score_partial = partial(batch_infill_criteria_score, batch=cobra['batch'], surrogateModels=surrogateModels, bestPredictor=bestPredictor, nObj=cobra['nObj'], infillCriteria=cobra['infillCriteria'], FresPlogStandardizedStd=cobra['FresPlogStandardizedStd'], FresStandardizedStd=cobra['FresStandardizedStd'], FresPlogStandardizedMean=cobra['FresPlogStandardizedMean'], FresStandardizedMean=cobra['FresStandardizedMean'], currentHV=cobra['currentHV'], paretoFrontier=cobra['paretoFrontier'], ref=cobra['ref'])

        f = partial(pool_job, criteria_function=compute_infill_criteria_score_partial, cons=cons, seqFeval = cobra['seqFeval'], seqTol=cobra['seqTol'])        
        if pool:
            submins = pool.map(f, xStarts)
        else:
            for xStart in xStarts:
                submins.append(f(xStart))

        for i in range(len(submins)):
            subMin = submins[i]
            success.append(subMin['success'])
            if subMin['fun'] < bestFun and subMin['success']:
                bestFun = subMin['fun']
                besti = i
        
        if all(success):
            minRequiredEvaluations = (cobra['dimension']+cobra['nConstraints']+cobra['nObj']+cobra['batch'])*20
            adjustedAmountEvaluations = int(cobra['seqFeval']*(1-cobra['surrogateUpdateLearningRate']))
            cobra['seqFeval'] = max(adjustedAmountEvaluations, minRequiredEvaluations)
            
            maxStartingPoints = (cobra['dimension']+cobra['nConstraints']+cobra['nObj'])*10*(1+int(cobra['batch']>1))*(1+int(cobra['oneShot']))
            adjustedAmountPoints = int(cobra['computeStartingPoints']*(1+cobra['surrogateUpdateLearningRate']))
            cobra['computeStartingPoints'] = min(maxStartingPoints, adjustedAmountPoints)
        else:
            maxRequiredEvaluations = (cobra['dimension']+cobra['nConstraints']+cobra['nObj'])*1000*(1+int(cobra['batch']>1))*(1+int(cobra['oneShot']))
            adjustedAmountEvaluations = int(cobra['seqFeval']*(1+cobra['surrogateUpdateLearningRate']))
            cobra['seqFeval'] = min(adjustedAmountEvaluations, maxRequiredEvaluations)
            
            minRequiredPoints = 2*(cobra['dimension']+cobra['nConstraints']+cobra['nObj']+cobra['batch'])
            adjustedAmountPoints = int(cobra['computeStartingPoints']*(1-cobra['surrogateUpdateLearningRate']))
            cobra['computeStartingPoints'] = max(adjustedAmountPoints, minRequiredPoints)
                
            
        if not any(success):
            print('NO SUCCESS', cobra['computeStartingPoints'], cobra['seqFeval'])
            smallest_constr = np.finfo(np.float64).min
            bestObj = np.finfo(np.float64).max
            besti = 0
            i = 0
            for subMin in submins:
                potX = subMin['x'] 
                potX = np.maximum(potX, list(cobra['lower'])*cobra['batch'])
                potX = np.minimum(potX, list(cobra['upper'])*cobra['batch'])
                potX = potX.reshape(cobra['batch'],int(len(subMin['x'])/cobra['batch']))
                potC = gCOBRA_partial(subMin['x'])
                cVoil = np.sum(potC[potC<0])
                
                notEvaluatedBefore = True
                for potXi in potX:
                    if potXi in cobra['A']:
                        notEvaluatedBefore=False
                        
                if notEvaluatedBefore and (cVoil > smallest_constr or (cVoil >= smallest_constr and subMin['fun'] < bestObj)):
                    smallest_constr = cVoil
                    bestObj = subMin['fun']
                    besti = i
                i += 1

        subMin = submins[besti]
        xNew = subMin['x']
        xNew = np.maximum(xNew, list(cobra['lower'])*cobra['batch'])
        xNew = np.minimum(xNew, list(cobra['upper'])*cobra['batch'])

        cobra['optimizerConvergence'] = np.append(cobra['optimizerConvergence'], subMin['status'])        
        return xNew   
    
    def get_best_predictor(cobra):
        if cobra['oneShot']:
            folds = 10
            surrogateErrors = {}
            for kernel in cobra['RBFmodel']:
                for obji in range(cobra['nObj']):
                    surrogateErrors['OBJ'+str(obji)+kernel] = []
                    surrogateErrors['OBJ'+str(obji)+'PLOG'+kernel] = []
                for coni in range(cobra['nConstraints']):
                    surrogateErrors['CON'+str(coni)+kernel] = []
                    surrogateErrors['CON'+str(coni)+'PLOG'+kernel] = []

            for foldi in range(folds):
                trainCobra = {'RBFmodel' : cobra['RBFmodel']}
                trainIndicator = np.array(range(len(cobra['A'])))
                trainCobra['A'] = cobra['A'][trainIndicator%folds!=foldi]
                trainCobra['FresStandardized'] = cobra['FresStandardized'][trainIndicator%folds!=foldi]
                trainCobra['FresPlogStandardized'] = cobra['FresPlogStandardized'][trainIndicator%folds!=foldi]
                trainCobra['GresRescaled'] = cobra['GresRescaled'][trainIndicator%folds!=foldi]
                trainCobra['GresPlogRescaled'] = cobra['GresPlogRescaled'][trainIndicator%folds!=foldi]
                surrogateModels = trainSurrogates(trainCobra)
                
                testCobra = {'RBFmodel' : cobra['RBFmodel']}
                testCobra['A'] = cobra['A'][trainIndicator%folds==foldi]
                testCobra['FresStandardized'] = cobra['FresStandardized'][trainIndicator%folds==foldi]
                testCobra['FresPlogStandardized'] = cobra['FresPlogStandardized'][trainIndicator%folds==foldi]
                testCobra['GresRescaled'] = cobra['GresRescaled'][trainIndicator%folds==foldi]
                testCobra['GresPlogRescaled'] = cobra['GresPlogRescaled'][trainIndicator%folds==foldi]
                testCobra['Fres'] = cobra['Fres'][trainIndicator%folds==foldi]
                testCobra['Gres'] = cobra['Gres'][trainIndicator%folds==foldi]
                
                for i in range(len(testCobra['A'])): 
                    yTrue = testCobra['Fres'][i]
                    conTrue = testCobra['Gres'][i]
                    x = testCobra['A'][i]
                    for obji in range(cobra['nObj']):
                        for kernel in cobra['RBFmodel']:
                            surrogatePlog = surrogateModels[kernel]['Objectives']['PLOGStandardized'][obji]
                            plogSol = interpRBF(x,surrogatePlog)
                            plogSol = plogSol*cobra['FresPlogStandardizedStd'][obji] + cobra['FresPlogStandardizedMean'][obji]
                            plogSol = plogReverse(plogSol)
                            surrogateErrors['OBJ'+str(obji)+'PLOG'+kernel].append((plogSol - yTrue[obji])**2)
                            
                            surrogate = surrogateModels[kernel]['Objectives']['Standardized'][obji]
                            sol = interpRBF(x, surrogate)
                            sol = sol*cobra['FresStandardizedStd'][obji] + cobra['FresStandardizedMean'][obji]
                            surrogateErrors['OBJ'+str(obji)+kernel].append((sol - yTrue[obji])**2)
                    
                    for coni in range(cobra['nConstraints']):
                        for kernel in cobra['RBFmodel']:
                            surrogatePlog = surrogateModels[kernel]['Constraints']['PLOGrescaled'][coni]
                            plogSol = interpRBF(x, surrogatePlog)
                            plogSol = cobra['GresPlogRescaledDivider'][coni] * plogSol
                            plogSol = plogReverse(plogSol)                    
                            surrogateErrors['CON'+str(coni)+'PLOG'+kernel].append((plogSol - conTrue[coni])**2)
                            
                            surrogate = surrogateModels[kernel]['Constraints']['Rescaled'][coni]
                            sol = interpRBF(x, surrogate)
                            sol = cobra['GresRescaledDivider'][coni] * sol
                            surrogateErrors['CON'+str(coni)+kernel].append((sol - conTrue[coni])**2)
            
            bestPredictor = {'objKernel':[], 'objLogStr':[], 'conKernel':[], 'conLogStr':[]}
            for obji in range(cobra['nObj']):
                minScore = np.finfo(np.float64).max
                bestPredictor['objKernel'].append(cobra['RBFmodel'][-1])
                bestPredictor['objLogStr'].append('Standardized')
                for kernel in cobra['RBFmodel']:
                    if np.sqrt(np.mean(np.array(surrogateErrors['OBJ'+str(obji)+kernel]))) < minScore:
                        minScore = np.sqrt(np.mean(np.array(surrogateErrors['OBJ'+str(obji)+kernel])))
                        bestPredictor['objKernel'][obji] = kernel
                        bestPredictor['objLogStr'][obji] = 'Standardized'
                    if np.sqrt(np.mean(np.array(surrogateErrors['OBJ'+str(obji)+'PLOG'+kernel]))) < minScore:
                        minScore = np.sqrt(np.mean(np.array(surrogateErrors['OBJ'+str(obji)+'PLOG'+kernel])))
                        bestPredictor['objKernel'][obji] = kernel
                        bestPredictor['objLogStr'][obji] = 'PLOGStandardized'
            for coni in range(cobra['nConstraints']):
                minScore = np.finfo(np.float64).max
                bestPredictor['conKernel'].append(cobra['RBFmodel'][-1])
                bestPredictor['conLogStr'].append('Rescaled')
                for kernel in cobra['RBFmodel']:
                    if np.sqrt(np.mean(np.array(surrogateErrors['CON'+str(coni)+kernel]))) < minScore:
                        minScore = np.sqrt(np.mean(np.array(surrogateErrors['CON'+str(coni)+kernel])))
                        bestPredictor['conKernel'][coni] = kernel
                        bestPredictor['conLogStr'][coni] = 'Rescaled'
                    if np.sqrt(np.mean(np.array(surrogateErrors['CON'+str(coni)+'PLOG'+kernel]))) < minScore:
                        minScore = np.sqrt(np.mean(np.array(surrogateErrors['CON'+str(coni)+'PLOG'+kernel])))
                        bestPredictor['conKernel'][coni] = kernel
                        bestPredictor['conLogStr'][coni] = 'PLOGrescaled'    
            return bestPredictor
            
        else:
            return cobra['bestPredictor'][-1]
    
    xNew = [None]
    yNewEval = [None]
    conNewEval = [None]
    surrogateModels = {} 

    while n < cobra['feval']:
        ptm = time.time()
        
        bestPredictor = get_best_predictor(cobra)
        
        surrogateModels = trainSurrogates(cobra)
        
        xNew = findSurrogateMinimum(cobra, surrogateModels, bestPredictor, pool)
        
        yNewEval = []
        conNewEval = []
        if cobra['batch'] == 1:
            yNew, cNew = fn(xNew)
            yNewEval.append(yNew)
            conNewEval.append(cNew)
            cobra = updateInfoAndCounters(cobra, xNew, yNew, cNew, phase, surrogateModels)
            xNew = np.array([xNew])
        else:
            xNew = xNew.reshape(cobra['batch'],int(len(xNew)/cobra['batch']))
            if pool:
                res = pool.map(fn, xNew)
            else:
                res = []
                for row in xNew:
                    res.append(fn(row))
            i = 0
            for result in res:
                yNew, cNew = result
                yNewEval.append(yNew)
                conNewEval.append(cNew)
                cobra = updateInfoAndCounters(cobra, xNew[i], yNew, cNew, phase, surrogateModels)
                i += 1
                
        n = len(cobra['A'])
        
        cobra['optimizationTime'] = np.append(cobra['optimizationTime'], time.time()-ptm)

        if cobra['plot']:
            print(time.time()-ptm, n, cobra['feval'], cobra['hypervolumeProgress'][-1], xNew)
            visualiseParetoFront(cobra['paretoFrontier'])
    
    functionName = str(fn).split(' ')[5]
    if cobra['oneShot']:
        outdir = 'oneshot/'+str(functionName)+'/'    
    else:
        outdir = 'batchresults/'+str(functionName)+'/batch'+str(cobra['batch'])+'/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    paretoOptimal = paretofrontFeasible(cobra['Fres'],cobra['Gres'])
    paretoFront = cobra['Fres'][paretoOptimal]
    paretoSet = cobra['A'][paretoOptimal]
    paretoConstraints = cobra['Gres'][paretoOptimal]
    runNo = cobra['cobraSeed']

    outputFileParameters = str(outdir)+'par_run'+str(runNo)+'_final.csv'
    outputFileObjectives = str(outdir)+'obj_run'+str(runNo)+'_final.csv'
    outputFileConstraints = str(outdir)+'con_run'+str(runNo)+'_final.csv'
    
    np.savetxt(outputFileParameters, cobra['A'], delimiter=',')
    np.savetxt(outputFileObjectives, cobra['Fres'], delimiter=',')
    np.savetxt(outputFileConstraints, cobra['Gres'], delimiter=',')

    outputFileParameters = str(outdir)+'par_run'+str(runNo)+'_finalPF.csv'
    outputFileObjectives = str(outdir)+'obj_run'+str(runNo)+'_finalPF.csv'
    outputFileConstraints = str(outdir)+'con_run'+str(runNo)+'_finalPF.csv'
    
    np.savetxt(outputFileObjectives, paretoFront, delimiter=',')
    np.savetxt(outputFileParameters, paretoSet, delimiter=',')
    np.savetxt(outputFileConstraints, paretoConstraints, delimiter=',')
    
    if pool:
        pool.close()
        pool.join()
    
    return(cobra)
