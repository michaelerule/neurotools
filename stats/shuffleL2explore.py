#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
"""
Statistical routines for shuffle-based hyothesis testing 
when using L2 regularized multivariate linear regression
to analyze explained variance.

These are useful when you have limited, highly correlated, 
highly nonlinear, highly skewed data, and don't trust an 
F test. They are very slow. 
"""

from neurotools.stats import partition_data_for_crossvalidation
from scipy.optimize   import minimize
import jax
import jax.numpy as jp
import jax.numpy.linalg as jl
from   jax import jit, grad
from   jax import jacfwd, jacrev
def hess(f,argnum=None):
    if argnum is None:
        return jacfwd(jacrev(f))
    return jacfwd(jacrev(f,argnum),)

import numpy as np
import scipy.stats
from typing import NamedTuple
from neurotools.util.time import progress_bar

############################################################
############################################################
############################################################
############################################################

@jit
def MSE(W,Σx,Σy,Σyx):
    Σyhat  = W@Σx@W.T
    Σyyhat = Σyx@W.T
    # ŷ = wx
    # <(y-ŷ)²> = trace( Σy + Σyhat - Σyyhat - Σyyhat.T )
    return jp.trace(Σy) + jp.trace(Σyhat) - 2*jp.trace(Σyyhat)

@jit
def error_covariance(W,Σx,Σy,Σyx):
    Σyh  = W@Σx@W.T
    Σyyh = Σyx@W.T
    return Σy + Σyh - Σyyh - Σyyh.T
    '''
    <(y-ŷ)(y-ŷ)'>
    <(y-ŷ)(y'-ŷ')>
    <(y(y'-ŷ')-ŷ(y'-ŷ'))>
    <yy'-yŷ'-ŷy'+ŷŷ'>
    <yy'>-<yŷ'>-<ŷy>'+<ŷŷ'>
    Σy + Σŷ - Σyŷ - Σyŷ'
    '''

############################################################
############################################################
############################################################
############################################################

@jit
def reglstq_moments(Σx,Σyx,rho):
    N   = Σx.shape[0]
    Σx += jax.numpy.eye(N)*rho
    W = Σyx @ jax.numpy.linalg.pinv(Σx)
    return W

@jit
def reglstsq_MSE(rho,Σx,Σyx,tΣx,tΣy,tΣyx):
    W = reglstq_moments(Σx,Σyx,rho)
    return MSE(W,tΣx,tΣy,tΣyx)
    
############################################################
############################################################
############################################################
############################################################

def regroup(x,y,nfold=10):
    return [*partition_data_for_crossvalidation(x,y,nfold,True)]

def ungroup(groups):
    _,_,x0,y0 = map(np.concatenate,zip(*groups))
    return x0,y0

@jit
def scov(X,Y):
    X = X-np.mean(X,0)
    Y = Y-np.mean(Y,0)
    return (X.T@Y)/(len(X)-1)

@jit
def xy_moments(trX,trY):
    Σx   = scov(trX,trX)
    Σy   = scov(trY,trY)
    Σyx  = scov(trY,trX)
    return Σx,Σy,Σyx
    
@jit
def reglstq_xy(x,y,rho):
    Σx,Σy,Σyx = xy_moments(x,y)
    N   = Σx.shape[0]
    Σx += jax.numpy.eye(N)*rho
    W = Σyx @ jax.numpy.linalg.pinv(Σx)
    return W

@jit
def group_moments(group):
    trX,trY,tsX,tsY = group
    Σx,Σy,Σyx = xy_moments(trX,trY)
    tΣx,tΣy,tΣyx = xy_moments(tsX,tsY)
    return Σx,Σy,Σyx,tΣx,tΣy,tΣyx
    
############################################################
############################################################
############################################################
############################################################

def group_mse(rho,groups):
    Σs = [group_moments(g) for g in groups]
    e2 = [reglstsq_MSE(rho,Σx,Σyx,tΣx,tΣy,tΣyx) 
          for Σx,Σy,Σyx,tΣx,tΣy,tΣyx in Σs]
    return np.mean(np.float32(e2))

#@jit
def group_error_covariance(rho,groups):
    E = []
    Σs = [*map(group_moments,groups)]
    for Σx,Σy,Σyx,tΣx,tΣy,tΣyx in Σs:
        W = reglstq_moments(Σx,Σyx,rho)
        E.append(error_covariance(W,tΣx,tΣy,tΣyx))
    return np.mean(E,0)

from scipy.linalg import cholesky as chol
from scipy.linalg.lapack import dtrtri
def get_whiten(S):
    # covariance transform
    # S = ULU'
    # x = Uq
    # q = inv(U)q
    U = chol(S,lower=True)   #q=UU'
    V = dtrtri(U,lower=1)[0] #inv(q)=V'V
    return V

def group_R2_matrix(rho,groups):
    '''
    Get matrix-valued R² for L2 regularized 
    (strength ``rho``)
    n-fold crossvalidated
    linear regression (n ``groups``).
    '''
    # Whitening transform from population covariance
    x0,y0 = ungroup(groups)
    S = scov(y0,y0)
    V = get_whiten(S)
    # Error covariances
    Σs = [*map(group_moments,groups)]
    E = group_error_covariance(rho,groups)
    I = np.eye(y0.shape[1])
    return I-V@E@V.T # R²
    
############################################################
############################################################
############################################################
############################################################
    
def regsweep_r2(groups,rr):
    '''
    Get list of R² for regularization strengths ``rr``.
    '''
    # Large regularization gives us chance level
    ibig = np.argmax(rr)
    rbig = np.max(rr)
    addedbig = False
    if rbig<1e9:
        rr = [*rr]+[1e9]
        ibig = len(rr)-1
        addedbig = True
    R2s = np.float32([
        group_R2_matrix(r,groups) for r in rr
    ])
    
    R2s -= R2s[ibig]
    if addedbig:
        R2s = R2s[:-1]
    return R2s
    

def regsweep_MSE(groups,rr):
    return np.float32([group_mse(r,groups) for r in rr])
    
############################################################
############################################################
############################################################
############################################################
  
def shuffle_variable(x,indeces):
    # Shuffle just some columns (``indeces``)
    # of ``x``
    N  = x.shape[0]
    i  = np.random.choice(np.arange(N),N,replace=False)
    nx = np.copy(x)
    for j in indeces:
        nx[:,j] = x[i,j]
    return nx  
    
def adaptive_shuffle(
    x,
    y,
    regularization_strengths,
    vg,
    nfold         = 10,
    minshuffles   = 100,
    maxshuffles   = 2000,
    tolerance     = 0.001,
    alpha         = 0.05,
    show_progress = True,
    return_samples= False,
    ):
    
    assert maxshuffles > minshuffles
    maxshuffles -= minshuffles
    
    # Large regularization gives us chance level
    rr  = sorted([*np.array(
        regularization_strengths).ravel()])
    if rr[-1]<1e9: rr += [1e9]
    
    # Get model fit with all features.
    groups = regroup(x,y,nfold)
    R0  = np.trace(regsweep_r2(groups,rr),axis1=1,axis2=2)
    R0 -= R0[-1]
    Rmax = np.max(R0)
    
    foo = progress_bar if show_progress else lambda x:x
    
    # Run a minimum number of shuffles
    R1 = []
    for i in foo(range(minshuffles)):
        R = regsweep_r2(regroup(shuffle_variable(x,vg),y,nfold),rr)
        R1.append(R)
        
    # Run shuffles until we have enough data for a clear p-value
    pvalue = None
    for i in foo(range(maxshuffles)):
        R = regsweep_r2(regroup(shuffle_variable(x,vg),y,nfold),rr)
        R1.append(R)
        
        # Check if we have enough results to make a clear decision
        u = np.trace(np.float32(R1),axis1=2,axis2=3)
        u-= u[:,-1][:,None]
        i_reg = np.argmax(np.percentile(u,50,axis=0))
        vuniq = Rmax-u[:,i_reg]
        
        k = np.sum(vuniq<0)
        n = len(vuniq)
        
        a,pvalue,b = scipy.stats.beta.ppf([
            tolerance/2,
            .5,
            1-tolerance/2],k+.5,n-k+.5)
        if show_progress:
            print('\r %s'%(
                scipy.stats.beta.ppf(
                    [.005,0.5,.995],k+.5,n-k+.5))+' '*10,
                end='',flush=True)
        if (a<alpha)==(b<alpha): 
            break
            
    if show_progress: print()
    result = ()
    if return_samples:
        return AdaptiveShuffleResult(
            pvalue, k, n, np.mean(vuniq), 
            Rmax, rr[i_reg], np.float32(R1))
    else:
        return AdaptiveShuffleResult(
            pvalue, k, n, np.mean(vuniq), 
            Rmax, rr[i_reg], None)

class AdaptiveShuffleResult(NamedTuple):
    pvalue: float
    k: int
    n: int
    vunique: float
    r2: float
    eta: float
    allr2: np.ndarray

def adaptive_shuffle_pvalue(
    sampling_function,
    relative_to    = 0.0,
    alternative    = 'two-sided', # 'two-sided', 'less', 'greater'
    minshuffles    = 10,
    maxshuffles    = 2000,
    ptol           = 0.01, # p density tolerance
    #atol           = 0.01, # absolute numerical tolerance
    alpha          = 0.05,
    show_progress  = True,
    return_samples = False,
    ):
    '''
    
    Parameters
    ----------
    sampling_function: function
        This function should take no arguments, and
        return a single scalar float reflecting the 
        value to be tested. 
    '''
    assert minshuffles > 0
    assert maxshuffles > minshuffles
    #assert atol>1e-15
    #assert atol<0.5
    assert ptol>1e-15
    assert ptol<0.1
    
    maxshuffles -= minshuffles
    alternative = str(alternative).lower()[0]
    
    samples = [sampling_function() 
               for i in range(minshuffles)]
    
    pvalue  = None
    for i in range(maxshuffles):
        samples.append(sampling_function())
        
        n = len(samples)
        u = np.float32(samples)
        k = np.sum(u<relative_to) # Greater
        
        # Bayesian Beta to get range on p-value
        a,b = k+.5,n-k+.5 
        lo,pvalue,hi = scipy.stats.beta.ppf([
            ptol/2,
            .5,
            1-ptol/2],a,b)
            
        v = a*b/((a+b)**2*(a+b+1))
        s = np.sqrt(v)
    
        if show_progress:
            print('\r %d %d %0.4f %0.4f %0.4f %0.4f'%(k,n,lo,pvalue,hi,s)+' '*10,
                end='',flush=True)
            
        # Absolute tolerance stop
        #if s<atol:
        # alternatives
        if alternative=='g':
            if (lo<alpha)==(hi<alpha): 
                break
        elif alternative=='l':
            if (lo>1-alpha)==(hi>1-alpha): 
                break
        else: #two-sided
            if (lo<alpha/2)==(hi<alpha/2): 
                break
            if (lo>1-alpha/2)==(hi>1-alpha/2): 
                break
    
    if show_progress: print()
    
    if alternative=='g':
        pass
    elif alternative=='l':
        pvalue = 1-pvalue
        k = n-k
    else: #two-sided
        if pvalue<0.5: # is bigger than
            pvalue*=2
        else: # is less than
            pvalue = 1-pvalue
            k = n-k
            
    result = (pvalue, k, n)
    if return_samples: result += (np.float32(samples),)
    return result

############################################################
############################################################
############################################################
############################################################

@jit
def joint_reglstsq_MSE(rho,Σs):
    N   = Σs[0][0].shape[0]
    R   = jax.numpy.eye(N)*rho
    err = 0.0
    for i,(Σx,_,Σyx,tΣx,_,tΣyx) in enumerate(Σs):
        W = Σyx @ jax.numpy.linalg.pinv(Σx + R)
        err += jp.sum((W.T@W)*tΣx) - 2*jp.sum(tΣyx*W)
    return err
    '''
    err = 0.0
    for Σx,Σy,Σyx,tΣx,tΣy,tΣyx in Σs:
        err += reglstsq_MSE(rho,Σx,Σyx,tΣx,tΣy,tΣyx)
    return err
    '''

def best_rho(Σx,Σyx,tΣx,tΣy,tΣyx,rho=0.1):
    @jit
    def o(rho):
        return reglstsq_MSE(rho,Σx,Σyx,tΣx,tΣy,tΣyx)
    g = jit(grad(o))
    h = jit(hess(o))
    result = minimize(o,rho,jac=g,hess=h,method='Newton-CG')
    return result.x

def joint_best_rho_gradient(groups,rho=1e6):
    # This routine peforms local gradient search on the 
    # regularization stranenggth, and can *usually*
    # gets trapped in a local minimum. 
    # don't use it!
    rho = np.log10(rho)
    nfold = len(groups)
    Σs = [*map(group_moments,groups)]
    N  = Σs[0][0].shape[0]
    @jit
    def o(rho):
        #'''
        R  = jax.numpy.eye(N)*(10.0**rho)
        err = 0.0
        for i,(Σx,_,Σyx,tΣx,_,tΣyx) in enumerate(Σs):
            W = Σyx @ jax.numpy.linalg.pinv(Σx + R)
            err += jp.sum((W.T@W)*tΣx) - 2*jp.sum(tΣyx*W)
        return err
        #'''
        return joint_reglstsq_MSE_reference(rho,Σs)
    g = jit(grad(o))
    h = jit(hess(o))
    result = minimize(o,rho,jac=g,hess=h,method='Newton-CG')
    return 10.**(result.x)



############################################################
############################################################
############################################################
############################################################



from numpy.random import *
from scipy.sparse.linalg import minres

def lfit(X,Y):
    Σx  = X@X.T
    Σxy = X@Y.T
    return minres(Σx, Σxy)[0].T

def leave_one_out(X,Y,rho=1e-6):
    '''
    Leave-one-out linear regression. 
    
    This is extremely inefficient for large datasets. 
    This was written for theoretical exploration only.
    It exists as a helper routine for 
    ``linear_regression_shuffle()``.
    
    This was written for theoretical exploration only.
    It is extremely inefficient and you should not use it.
    
    Parameters
    ----------
    X: Ncovariates × Nsames np.ndarray
        Array of values for the independent variables
    Y: Ndependent × Nsamples np.ndarray
        Array of values for the dependent variables
    rho: positive float; default 1e-6
        L2 regularization
    
    Returns
    -------
    Yhat: np.ndarray
        Leave-one-out predictions of the dependent 
        variables.
    
    '''    
    N,T = X.shape
    X   = add_constant(X,axis=0)
    R = np.eye(N+1)*rho
    R[-1,-1] = 0
    Σx  = X@X.T + R
    Σxy = X@Y.T
    Yhat = []
    for t in range(T):
        Σx_  = Σx  - outer(X[:,t],X[:,t])
        Σxy_ = Σxy - outer(X[:,t],Y[:,t])
        W = minres(Σx_, Σxy_)[0].T
        Yhat.append(W@X[:,t])
    return array(Yhat)

def linear_regression_shuffle(
    X,
    Y, 
    nshuffle = 100,
    rho = 1e-6):
    '''
    Leave-one-out linear regression with shuffle-out
    tests for individual parameter importance. 
    
    This was written for theoretical exploration only.
    It is extremely inefficient and you should not use it.
    
    Parameters
    ----------
    X: Ncovariates × Nsames np.ndarray
        Array of values for the independent variables
    Y: Ndependent × Nsamples np.ndarray
        Array of values for the dependent variables
    NSHUFFLE: positive int; default 100
        Number of shuffles to sample
    rho: positive float; default 1e-6
        L2 regularization
    
    Returns
    -------
    yhat: np.float32
        Leave-one-out predicted values for the dependent 
        variable.
    mse: float
        Mean-squared-error (MSE) for leave-one-out linear
        regression.
    shuffles: Ncovariates × Nshuffles np.ndarray
        List of MSE for shuffle-out tests for all
        covariates. 
    '''
    N,T = X.shape
    
    mse = lambda x,y: mean((x-y)**2)
    r2  = lambda x,y: 1 - mse(x,y)/var(y)
    rho*= T
    MSE = [[] for n in range(N)]
    for n in range(N):
        X_ = array(X)
        for i in range(nshuffle):
            X_[n,:] = choice(X[n,:],T,replace=False)
            Yhat = leave_one_out(X_,Y,rho=rho)
            MSE[n].append(mse(Y,Yhat))
    
    Yhat = leave_one_out(X,Y,rho=rho)
    mse_ = mse(Y,Yhat)
    
    return Yhat, mse_, array(MSE)
