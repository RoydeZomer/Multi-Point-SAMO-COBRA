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

def SAMO_COBRA_PhaseII(cobra):

    # print("PHASE II started")
    phase = 'PHASE II'
    if cobra['hypervolumeProgress'] is None:
        raise ValueError("cobraPhaseII: cobra['hypervolumeProgress'] is None! First run smscobraInit")
        
    fn = cobra['fn']
    cobra['EPS'] = np.array(cobra['epsilonInit']) # Initializing margin for all constraints
    n = len(cobra['A'])
    if n==cobra['initDesPoints']:
        predHV = np.empty(cobra['initDesPoints'])
        predHV[:] = np.nan # structure to store surrogate optimization results
        if cobra['nConstraints']!=0:
            cobra['predC'] = np.empty((cobra['initDesPoints'], cobra['nConstraints'])) # matrix to store predicted constraint values
            cobra['predC'][:] = 0
        cobra['optimizerConvergence'] = np.ones(cobra['initDesPoints']) # vector to store optimizer convergence
        feval = np.empty(cobra['initDesPoints'])
        feval[:] = np.nan
        
    if n >= cobra['feval']:
        raise ValueError("ERROR! Number of function evaluations after initialization is larger than total allowed evaluations")
        
    def updateInfoAndCounters(cobra, xNew, yNewEval, conNewEval, phase):
        
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
                cobra['EPS'][ci] = cobra['EPS'][ci]*0.9
            else:
                cobra['EPS'][ci] = np.minimum(1.1*cobra['EPS'][ci],cobra['epsilonMax'][ci])
                
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
            
    def subSMSProb2(x):
        nonlocal surrogateModels
        nonlocal bestPredictor  
        nonlocal cobra
        # global surrogateModels
        # global bestPredictor  
        # global cobra
        
        if np.any(np.isnan(x)):
            return np.finfo(np.float64).max
        if not all(np.isfinite(x)):
            return np.finfo(np.float64).max

        potentialSolution = np.zeros(cobra['nObj'])
        for obji in range(cobra['nObj']):
            objKernel = bestPredictor['objKernel'][obji]
            objLogStr = bestPredictor['objLogStr'][obji]
            surrogate = surrogateModels[objKernel]['Objectives'][objLogStr][obji]
            
            potsol = 0
            uncertainty = 0
            if objLogStr=='PLOGStandardized':
                if cobra['infillCriteria'] == 'PHV':
                    potsol = interpRBF(x, surrogate, uncertainty=False)                
                    potsol = potsol*cobra['FresPlogStandardizedStd'][obji] + cobra['FresPlogStandardizedMean'][obji]
                    potsol = plogReverse(potsol)
                elif cobra['infillCriteria'] == 'SMS':
                    potsol, uncertainty = interpRBF(x, surrogate, uncertainty=True)
                    uncertainty = uncertainty * cobra['FresStandardizedStd'][obji]
                    potsol = potsol*cobra['FresPlogStandardizedStd'][obji] + cobra['FresPlogStandardizedMean'][obji]
                    potsol = plogReverse(potsol)
                else:
                    raise ValueError("This infill criteria is not implemented")
            else:
                if cobra['infillCriteria'] == 'PHV':
                    potsol = interpRBF(x, surrogate, uncertainty=False)
                    potsol = potsol*cobra['FresStandardizedStd'][obji] + cobra['FresStandardizedMean'][obji]
                elif cobra['infillCriteria'] == 'SMS':
                    potsol, uncertainty = interpRBF(x, surrogate, uncertainty=True)
                    potsol = potsol*cobra['FresStandardizedStd'][obji] + cobra['FresStandardizedMean'][obji]
                    uncertainty = uncertainty * cobra['FresStandardizedStd'][obji]
                else:
                    raise ValueError("This infill criteria is not implemented")
                                
            if not np.isfinite(potsol):
                return np.finfo(np.float64).max
            if not np.isfinite(uncertainty):
                return np.finfo(np.float64).max
            
            potentialSolution[obji] = potsol - np.abs(uncertainty)
                
        currentHV = cobra['currentHV']
        paretoFront = cobra['paretoFrontier']
        ref = cobra['ref']
        
        penalty = 0
        ##### add epsilon?
        if not all(np.isfinite(potentialSolution)):
            return np.finfo(np.float64).max
        
        logicBool = np.all(paretoFront<= potentialSolution, axis=1)
        for j in range(paretoFront.shape[0]):
            if logicBool[j]:
                p = - 1 + np.prod(1 + (potentialSolution-paretoFront[j,:]))
                penalty = max(penalty, p)
        if penalty == 0: #non-dominated solutions
            potentialFront = np.append(paretoFront, [potentialSolution], axis=0)
            myhv = hypervolume(potentialFront, ref)
            f = currentHV - myhv
        else:
            f = penalty
        return f
    
    def getConstraintPrediction(x, EPS=None):
        nonlocal surrogateModels
        nonlocal bestPredictor  
        nonlocal cobra
        # global cobra
        # global surrogateModels
        # global bestPredictor

        constraintPredictions = np.zeros(cobra['nConstraints'])
        for coni in range(cobra['nConstraints']):
            conKernel = bestPredictor['conKernel'][coni]
            conLogStr = bestPredictor['conLogStr'][coni]
            surrogate = surrogateModels[conKernel]['Constraints'][conLogStr][coni]
            
            if conLogStr=='PLOGrescaled':
                constraintPrediction = interpRBF(np.array(x), surrogate)
                constraintPrediction = cobra['GresPlogRescaledDivider'][coni] * constraintPrediction
                constraintPrediction = plogReverse(constraintPrediction)
            else:
                constraintPrediction = interpRBF(np.array(x), surrogate)
                constraintPrediction = cobra['GresRescaledDivider'][coni] * constraintPrediction                
            
            if EPS is None:
                constraintPredictions[coni] = constraintPrediction
            else:
                constraintPredictions[coni] = constraintPrediction+EPS[coni]**2

        return constraintPredictions
    
    def gCOBRA(x):
        nonlocal cobra
        # global cobra
        h = 0
        distance = distLine(x, cobra['A'])
        if any(distance<=(len(cobra['A'][0])/1e4)):
            if min(distance) != 0:
                h = min(distance)**-2
            else:
                h = np.finfo(np.float64).max
        if not all(np.isfinite(distance)):
            h = np.finfo(np.float64).max
            
        constraintPrediction = getConstraintPrediction(x, cobra['EPS'])
        
        if np.any(np.isnan(constraintPrediction)):
            warnings.warn('gCOBRA: constraintPrediction value is NaN, returning Inf',DeprecationWarning)
            return([np.finfo(np.float64).min]*(len(constraintPrediction)+1))
        
        h = np.append(np.array([-1*h]), -1*constraintPrediction) #cobyla treats positive values as feasible
        return(h)

    def computeStartPoints(cobra):
        np.random.seed(cobra['cobraSeed']+len(cobra['A'])+1)
        lb = cobra['lower']
        ub = cobra['upper']
        strategy = cobra['computeStartPointsStrategy']
        
        if strategy=='random':
            startPoints = lb + np.random.rand(len(lb)) * (ub-lb)
        
        elif strategy=='multirandom':
            numberStartPoints = cobra['computeStartingPoints']
            startPoints = np.random.rand(len(lb)*numberStartPoints)
            startPoints = startPoints.reshape((numberStartPoints,len(lb)))
            startPoints = lb + startPoints * (ub-lb)
        
        elif strategy=='LHS':
            numberStartPoints = cobra['computeStartingPoints']
            startPoints = lhs(len(lb), samples=numberStartPoints, criterion="center", iterations=5)
            
        elif strategy=='midle':
            startPoints = (lb + ub)/ 2
        else:
            # do something smart?
            raise ValueError("This strategy does not exist for computeStartPoints")
            
        return startPoints
    
    def findSurrogateMinimum(cobra, surrogateModels, bestPredictor):
        xStarts = computeStartPoints(cobra)
        cons = []
        cons.append({'type':'ineq','fun':gCOBRA})        
        
        for factor in range(len(cobra['lower'])):
            lower = cobra['lower'][factor]
            l = {'type':'ineq','fun': lambda x, lb=lower, i=factor: x[i]-lb}
            cons.append(l)
        for factor in range(len(cobra['upper'])):
            upper = cobra['upper'][factor]
            u = {'type':'ineq','fun': lambda x, ub=upper, i=factor: ub-x[i]}
            cons.append(u)
        
        submins = []
        besti = 0
        bestFun = 0
        success = []
        
        for i in range(len(xStarts)):
            xStart = xStarts[i]
            opts = {'maxiter':cobra['seqFeval'], 'tol':cobra['seqTol']}
            subMin = optimize.minimize(subSMSProb2, xStart, constraints=cons, options=opts, method='COBYLA')
            submins.append(subMin)
            success.append(subMin['success'])            
            if subMin['fun'] < bestFun and subMin['success']:
                bestFun = subMin['fun']
                besti = i
        
        
        if all(success):
            minRequiredEvaluations = (cobra['dimension']+cobra['nConstraints']+cobra['nObj'])*20
            adjustedAmountEvaluations = int(cobra['seqFeval']*0.9)
            cobra['seqFeval'] = max(adjustedAmountEvaluations, minRequiredEvaluations)
            
            maxStartingPoints = (cobra['dimension']+cobra['nConstraints']+cobra['nObj'])*10
            adjustedAmountPoints = int(cobra['computeStartingPoints']*1.1)
            cobra['computeStartingPoints'] = min(maxStartingPoints, adjustedAmountPoints)
        else:
            maxRequiredEvaluations = (cobra['dimension']+cobra['nConstraints']+cobra['nObj'])*1000
            adjustedAmountEvaluations = int(cobra['seqFeval']*1.1)
            cobra['seqFeval'] = min(adjustedAmountEvaluations, maxRequiredEvaluations)
            
            minRequiredPoints = 2*(cobra['dimension']+cobra['nConstraints']+cobra['nObj'])
            adjustedAmountPoints = int(cobra['computeStartingPoints']*0.9)
            cobra['computeStartingPoints'] = max(adjustedAmountPoints, minRequiredPoints)
                
            
        if not any(success):
            print('NO SUCCESS', cobra['computeStartingPoints'], cobra['seqFeval'])
            smallest_constr = np.finfo(np.float64).max
            bestObj = np.finfo(np.float64).max
            besti = 0
            i = 0
            for subMin in submins:
                if subMin['maxcv'] < smallest_constr or (subMin['maxcv']<= smallest_constr and subMin['fun'] < bestObj):
                    smallest_constr = subMin['maxcv']
                    bestObj = subMin['fun']
                    besti = i
                i += 1

        subMin = submins[besti]
        xNew = subMin['x']
        xNew = np.maximum(xNew, cobra['lower'])
        xNew = np.minimum(xNew, cobra['upper'])

        cobra['optimizerConvergence'] = np.append(cobra['optimizerConvergence'], subMin['status'])        
        return xNew

    def define_best_predictor(x, yTrue, conTrue, surrogateModels, cobra): 
        if xNew is not None: # check if dict is empty in first iteration.
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
        hvgrowing[-4:] = True # besides the hypervolume improvement iterations, the last 4 results also count!
        
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
        return bestPredictor
        

    xNew = None
    yNewEval = None
    conNewEval = None
    surrogateModels = {} 

    while n < cobra['feval']:
        ptm = time.time()

        bestPredictor = define_best_predictor(xNew, yNewEval, conNewEval, surrogateModels, cobra)
        
        surrogateModels = trainSurrogates(cobra)
        
        xNew = findSurrogateMinimum(cobra, surrogateModels, bestPredictor)
        yNewEval, conNewEval = fn(xNew)
        
        cobra['predC'] = np.vstack((cobra['predC'], getConstraintPrediction(xNew)))
        cobra = updateInfoAndCounters(cobra, xNew, yNewEval, conNewEval, phase)
                
        n = len(cobra['A'])
        
        cobra['optimizationTime'] = np.append(cobra['optimizationTime'], time.time()-ptm)

        if cobra['plot']:
            print(time.time()-ptm, n, cobra['feval'], cobra['hypervolumeProgress'][-1], xNew, min(distLine(xNew, cobra['A'])))
            # visualiseParetoFront(cobra['paretoFrontier'])
    
    # functionName = cobra['fName']
    # outdir = 'results/'+str(functionName)+'/'        
    # if not os.path.isdir(outdir):
    #     os.makedirs(outdir)
    
    # paretoOptimal = paretofrontFeasible(cobra['Fres'],cobra['Gres'])
    # paretoFront = cobra['Fres'][paretoOptimal]
    # paretoSet = cobra['A'][paretoOptimal]
    # paretoConstraints = cobra['Gres'][paretoOptimal]
    # runNo = cobra['cobraSeed']

    # outputFileParameters = str(outdir)+'par_run'+str(runNo)+'_final.csv'
    # outputFileObjectives = str(outdir)+'obj_run'+str(runNo)+'_final.csv'
    # outputFileConstraints = str(outdir)+'con_run'+str(runNo)+'_final.csv'
    
    # np.savetxt(outputFileParameters, cobra['A'], delimiter=',')
    # np.savetxt(outputFileObjectives, cobra['Fres'], delimiter=',')
    # np.savetxt(outputFileConstraints, cobra['Gres'], delimiter=',')

    # outputFileParameters = str(outdir)+'par_run'+str(runNo)+'_finalPF.csv'
    # outputFileObjectives = str(outdir)+'obj_run'+str(runNo)+'_finalPF.csv'
    # outputFileConstraints = str(outdir)+'con_run'+str(runNo)+'_finalPF.csv'
    
    # np.savetxt(outputFileObjectives, paretoFront, delimiter=',')
    # np.savetxt(outputFileParameters, paretoSet, delimiter=',')
    # np.savetxt(outputFileConstraints, paretoConstraints, delimiter=',')
    
    return(cobra)
