# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:26:17 2017

@author: r.dewinter
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calcRHS(U,d2):
    if d2 is not None:
        if U.ndim == 1:
            rhs = np.append(U,np.zeros(d2))
        elif U.ndim == 2:
            rhs = np.vstack((U,np.array([d2*[0]]).T))
        else:
            raise ValueError('U is neither vector nor matirx!')
    else:
        rhs = U
    return rhs

def svdInv(M):
    eps = 1E-14
    M = np.nan_to_num(M)
    u,s,v = np.linalg.svd(M)
    invD = 1/s
    invD[abs(s/s[0])<eps] = 0
    invM = np.matmul(v.T,np.matmul(np.diag(invD),u.T))
    return(invM)

def fitRBF(phi, U, ptail=True, squares=False, xp=None, rbftype="THINPLATESPLINE", width=1):
    '''
    Fit  RBF interpolation to training data for d>1.
    
    The model for a point z=(z_1,...,z_d) is fitted using n sample points x_1, ..., x_n 
       s(z) = \lambda_1*\Phi(||z-x_1||)+... +\lambda_n*\Phi(||z-x_n||)
                     + c_0 + c_1*z_1 + ... + c_d*z_d  
    
    where \Phi(r)=r^3 denotes the cubic radial basis function. The coefficients \lambda_1, 
    ..., \lambda_n, c_0, c_1, ..., c_d are determined by this training procedure.
    This is for the default case squares==FALSE. In case squares==TRUE 
    there are d additional pure square terms and the model is
    
       s_sq(z) = s(z) + c_d+1*z_1^2 + ... + c_d+d*z_d^2  
    
      
    The linear equation system is solved via SVD inversion. Near-zero elements 
    in the diagonal matrix D are set to zero in D^-1. This is numerically stable 
    for rank-deficient systems.
    
    @param xp      n points x_i of dimension d are arranged in (n x d) matrix xp
    @param U       vector of length n, containing samples u(x_i) of 
                   the scalar function u to be fitted 
                   - or - 
                   (n x m) matrix, where each column 1,...,m contains one vector of samples
                   u_j(x_i) for the m'th model, j=1,...,m
    @param squares [FALSE] flag, see description
                   
    @return rbf.model,  an object of class RBFinter, which is basically a list 
    with elements:
         coef  (n+d+1 x m) matrix holding in column m the coefficients for the m'th 
                       model:      \lambda_1, ..., \lambda_n, c_0, c_1, ..., c_d.  
                       In case squares==TRUE it is an (n+2d+1 x m) matrix holding  
                       additionally the coefficients c_d+1, ..., c_d+d.                    
         xp  matrix xp   
         d  dimension d 
         npts  number n of points x_i 
         squares  TRUE or FALSE  
         type  "CUBIC"
         
    @seealso   trainGaussRBF, predict.RBFinter, interpRBF
    '''    
    npts = len(xp)
    d2 = None
    pMat = None
    
    if not ptail and not squares:
        M = phi
    else:
        if ptail:
            #linear tail
            e = np.array(npts*[1.0])
            pMat = np.column_stack((e,xp))
        if squares:
            # plus direct squares x1**2 x2**2,....
            if pMat is not None:
                pMat = np.column_stack((pMat,xp*xp))
            else:
                pMat = xp*xp
    
        d2 = len(pMat[0])
        nMat = np.zeros((d2,d2))
        M = np.column_stack((phi, pMat))
        
        QQ = pMat.transpose()
        M = np.vstack((M,np.column_stack((QQ,nMat))))
    
    invM = svdInv(M)
    rhs = calcRHS(U,d2)
    coef = np.matmul(invM,rhs)
    
    rbfmodel = dict()
    rbfmodel['coef'] = coef
    rbfmodel['xp'] = xp
    rbfmodel['d0'] = len(xp[0])
    rbfmodel['d'] = d2
    rbfmodel['npts'] = npts
    rbfmodel['ptail'] = ptail
    rbfmodel['squares'] = squares
    rbfmodel['type'] = rbftype
    rbfmodel['width'] = width
    
    #computations for for uncertainty quantification
    # phiinv = svdInv(phi)
    # rbfmodel['phiinv'] = phiinv
    rbfmodel['phiinv'] = invM
    return rbfmodel

def trainRBF(xp, U, ptail=True, squares=False, smooth=0.001, rbftype="THINPLATESPLINE"):
    edist = euclidean_distances(xp)
    # edist = edist + np.eye(len(xp))*smooth
    width = 1
    if rbftype=='CUBIC':
        phi = edist*edist*edist
    elif rbftype=='THINPLATESPLINE':
        phi = edist*edist*np.log(edist, where=edist!=0)
        phi = np.nan_to_num(phi)
    elif rbftype=='POLYHARMONIC1':
        phi = edist
    elif rbftype=='POLYHARMONIC4':
        phi = edist*edist*edist*edist*np.log(edist, where=edist!=0)
        phi = np.nan_to_num(phi)
    elif rbftype=='POLYHARMONIC5':
        phi = edist*edist*edist*edist*edist
    elif rbftype=='MULTIQUADRIC':
        # width = np.max(edist)/np.sqrt(2*len(xp))
        # width = np.max(xp) - np.min(xp)
        phi = (1+(width*edist)*(width*edist))**0.5
    elif rbftype=='GAUSSIAN':
        # width = np.max(edist)/np.sqrt(2*len(xp))
        # width = np.max(xp) - np.min(xp)
        phi = np.exp(-1*((width*edist)*(width*edist)))
    elif rbftype=='INVMULTIQUADRIC':
        # width = np.max(edist)/np.sqrt(2*len(xp))
        # width = np.max(xp) - np.min(xp)
        phi = 1/(1+(width*edist)*(width*edist))**0.5
    elif rbftype=='INVQUADRIC':
        # width = np.max(edist)/np.sqrt(2*len(xp))
        # width = np.max(xp) - np.min(xp)
        phi = 1 / (1+((width*edist)*(width*edist)))
    else:
        raise ValueError('RBF TYPE NOT IMPLEMENTED')
        
    phi = phi + np.eye(len(xp))*smooth
    rbfmodel = fitRBF(phi,U,ptail,squares,xp,rbftype=rbftype,width=width)
    return rbfmodel


def distLine(x, xp):
    z = x - xp
    zr = np.sqrt(np.sum(z*z,axis=1))
    return zr

def interpRBF(x, rbfModel, uncertainty=False):
    '''
    Apply cubic or Gaussian RBF interpolation to new data for d>1.
    
    param x         vector holding a point of dimension d
    param rbf.model trained RBF model (or set of models), see trainCubicRBF
                     or trainGaussRBF
                   
    return          value s(x) of the trained smodel at x
                     - or - 
                     vector s_j(x) with values for all trained models j=1,...,m at x
    
    seealso   trainCubicRBF, predict.RBFinter
    '''
    if len(x)!=rbfModel['d0']:
        print(x)
        print(rbfModel['xp'])
        raise ValueError('Problem in interpRBF, length of vector and rbf model do not match!')
    
    ed = distLine(x, rbfModel['xp']) # euclidean distance of x to all xp, ed is vector of length nrow(xp)  
    
    width = rbfModel['width']
    if rbfModel['type']=='CUBIC':
        ph = ed*ed*ed
    elif rbfModel['type']=='THINPLATESPLINE':
        ph = ed*ed*np.log(ed,where=ed!=0)
        # ph = np.nan_to_num(ph)
    elif rbfModel['type']=='POLYHARMONIC1':
        ph = ed
    elif rbfModel['type']=='POLYHARMONIC4':
        ph = ed*ed*ed*ed*np.log(ed, where=ed!=0)
    elif rbfModel['type']=='POLYHARMONIC5':
        ph = ed*ed*ed*ed*ed
    elif rbfModel['type']=='MULTIQUADRIC':
        ph = (1+(width*ed)*(width*ed))**0.5
    elif rbfModel['type']=='GAUSSIAN':
        ph = np.exp(-1*(width*ed)*(width*ed))
    elif rbfModel['type']=='INVMULTIQUADRIC':
        ph = 1/(1+(width*ed)*(width*ed))**0.5
    elif rbfModel['type']=='INVQUADRIC':
        ph = 1 / (1+(width*ed)*(width*ed))
    else:
        raise ValueError('RBF TYPE NOT IMPLEMENTED')
        
    if rbfModel['ptail']:
        if rbfModel['squares']:
            lhs = np.empty(len(ph)+1+len(x)*2)
            lhs[:len(ph)] = ph
            lhs[len(ph)]=1
            lhs[len(ph)+1:len(ph)+1+len(x)] = x
            lhs[len(ph)+1+len(x):] = np.multiply(x,x)
            # lhs = np.append(ph, 1)
            # lhs = np.append(lhs, x)
            # lhs = np.append(lhs, np.multiply(x,x))
        else:
            lhs = np.append(ph,1)
            lhs = np.append(lhs, x)
    else:
        lhs = ph
    
    val = np.matmul(lhs,rbfModel['coef']).item()
    
    if uncertainty:
        phiinv = rbfModel['phiinv']
        uncertainty_value=0
        if rbfModel['type']=='CUBIC':
            uncertainty_value = 0 - np.matmul(np.matmul(lhs,phiinv),lhs.T)
        elif rbfModel['type']=='THINPLATESPLINE':
            uncertainty_value = 0 - np.matmul(np.matmul(lhs,phiinv),lhs.T)
        elif rbfModel['type']=='MULTIQUADRIC':
            uncertainty_value = 1 - np.matmul(np.matmul(lhs,phiinv),lhs.T)
        elif rbfModel['type']=='GAUSSIAN':
            uncertainty_value = 1 - np.matmul(np.matmul(lhs,phiinv),lhs.T)
        elif rbfModel['type']=='INVMULTIQUADRIC':
            uncertainty_value = 1 - np.matmul(np.matmul(lhs,phiinv),lhs.T)
        elif rbfModel['type']=='INVQUADRIC':
            uncertainty_value = 1 - np.matmul(np.matmul(lhs,phiinv),lhs.T)
        else:
            raise ValueError('RBF TYPE NOT IMPLEMENTED')
        
        if np.isfinite(val): 
            return val, np.abs(uncertainty_value)
        else:
            return np.inf, np.abs(uncertainty_value)

    else:
        if np.isfinite(val): 
            return val
        else:
            return np.inf

def predictRBFinter(rbfModel, newdata, uncertainty=False):
    '''
    Apply cubic or Gaussian RBF interpolation
     
    Apply cubic or Gaussian RBF interpolation to a set of new data points for d>1.
    
    param rbf.model trained RBF model (or set of models), see trainCubicRBF 
                     or trainGaussRBF
    param newdata   matrix or data frame with d columns. Each row contains a data point 
                      x_i, i=1,...,n
                    
    return          vector of model responses s(x_i), one element for each data point x_i
                     - or - 
                     if rbf.model is a set of m models, a (n x m)-matrix 
                      containing in each row the response s_j(x_i) of all models 
                     j = 1,...,m to x_i
     
    seealso   trainCubicRBF, trainGaussRBF, interpRBF
    '''
    val = [interpRBF(i,rbfModel, uncertainty) for i in newdata]
    return(val)