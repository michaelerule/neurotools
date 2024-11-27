#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Tutorial in Poisson generalized linear point-process models
for neural spike train analysis.

Depends on

 - [numpy and scipy](http://www.scipy.org/install.html)
 - [sklearn](http://scikit-learn.org/stable/install.html)
 - [statsmodels](http://statsmodels.sourceforge.net/devel/install.html)
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

#############################################################################
# Imports
# Check that numpy and scipy is installed
try:
    import numpy as np
except:
    print('no numpy; glm routines will not work')

# get GLM solver from statsmodels
try:
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod.families import Poisson
except:
    print('no statsmodels; glm routines will not work')

# get ROC curve code from sklearn
try:
    # This AUC algorithm is not the best, but it will do
    from sklearn.metrics import roc_auc_score as auc
except:
    print('no sklearn; ROC curve routines missing')

# the function minimize wraps a large number of function optimizers
try:
    from scipy.optimize import minimize
except:
    print('no scipy; glm routines will not work')

import warnings

#############################################################################

def GLMPenaltyPoisson(X,Y):
    '''
    Generates objective, gradient, and hessian functions for the Poisson 
    GLM for design matrix X and spike observations Y.
    
    Parameters
    ----------
    X: array-like
        N observation x D features design matrix
    Y: array-like
        N x 1 point-process counts

    Returns
    -------
    objective: function
        objective function for `scipy.optimize.minimize`
    gradient: function
        gradient (jacobian) function for `scipy.optimize.minimize`
    hessian: function
        hessian function for `scipy.optimize.minimize`
    '''
    X = cat([ones((X.shape[0],1)),X],axis=1)
    N,D = X.shape
    Y = np.squeeze(Y)
    X = np.float64(X)         # use double precision
    K = np.sum(Y)             # total number of events
    Z = np.sum(X*Y[:,None],0) # event-conditioned sums of X
    scale = 1./N              # normalized by the amount of data. can be tweaked?
    def objective(H):
        rate = np.exp(X@H)
        like = Z@H-np.sum(rate)
        return -like*scale
    def gradient(H):
        rate = np.exp(X@H)
        grad = Z.T - X.T @ rate
        return -grad*scale
    def hessian(H):
        rate = np.exp(X@H)
        hess = X.T@(rate[:,None]*X)
        return hess*scale
    return objective, gradient, hessian

def sexp(x):
    '''
    Exponential function avoiding overflow and underflow, for evaluating the
    logistic sigmoid function
    '''
    x = np.clip(x,-708,708)
    return np.exp(x)

def sigmoid(x):
    '''
    Logistic sigmoid function
    '''
    s = 1/(1+sexp(-x))
    if type(s) is np.ndarray:
        s[s<1e-300]=0
    else:
        if s<1e-300:
            s=0
    return s

def GLMPenaltyBernoulli(X,Y):
    '''
    Generates objective, gradient, and hessian functions for the Bernoulli 
    GLM for design matrix X and spike observations Y.
    
    Parameters
    ----------
    X: array-like
        N observation x D features design matrix
    Y: array-like
        N x 1 point-process counts

    Returns
    -------
    objective: function
        objective function for `scipy.optimize.minimize`
    gradient: function
        gradient (jacobian) function for `scipy.optimize.minimize`
    hessian: function
        hessian function for `scipy.optimize.minimize`
    '''
    X = cat([ones((X.shape[0],1)),X],axis=1)
    N,D = X.shape
    Y = np.squeeze(Y)
    X = np.float64(X) # use double precision
    scale = 1./N      # normalized by the amount of data. can be tweaked?
    def objective(H):
        rate = sigmoid(X @ H)
        lnPr = Y * -log(1+sexp(-rate)) + (1-Y) * -log(1+sexp(rate))
        return -np.sum(lnPr)*scale
    def gradient(H):
        rate = sigmoid(X@H)
        grad = np.sum((Y - rate)[:,None]*X,axis=0)
        return -grad*scale
    def hessian(H):
        rate  = sigmoid(X@H)
        slope = rate*(1-rate)
        hess  = -X.T@(slope[:,None]*X)
        return -hess*scale
    return objective, gradient, hessian

def GLMPenaltyL2(X,Y,penalties=None):
    '''
    Generates objective, gradient, and hessian functions for the penalized
    L2 regularized poisson GLM for design matrix X and spike observations Y.

    Parameters
    ----------
    X: array-like
        N observation x D features design matrix
    Y: array-like
        N x 1 point-process counts
    penalties: 
        len D-1 list of L2 penalty weights (don't penalize mean)

    Returns
    -------
    objective: function
        objective function for `scipy.optimize.minimize`
    gradient: function
        gradient (jacobian) function for `scipy.optimize.minimize`
    hessian: function
        hessian function for `scipy.optimize.minimize`
    '''
    N,D = X.shape
    if N<D:
        warnings.warn('# samples < # features; is input transposed?')
    #print('@',penalties)
    if penalties is None: 
        penalties = np.zeros((D,),'d')
    if type(penalties) in (float,int) or np.prod(np.shape(penalties))==1:
        #print('Penalty parameter is a scalar')
        #print('Penalizing all parameters with α=',penalties)
        penalties = np.array(penalties).ravel()[0]
        penalties = np.ones((D,),dtype='d')*penalties
    #print(penalties)
    if not np.shape(penalties)==(D,):
        raise ValueError('Regularization penalty, if provided, should eiher be a scalar or a vector of one penalty per fecture.');
    Y = np.squeeze(Y)
    if not Y.shape==(N,) and len(Y.shape)==1:
        raise ValueError('Y should consist of a single binary vector with the same number of samples as the features.');
    #Y = np.int32(Y>0.5)     # binarize    
    X = np.float64(X)       # use double precision
    K = np.sum(Y)           # total number of events
    Z = np.sum(X*Y[:,None],0) # event-conditioned sums of X
    scale = 1./N         # normalized by the amount of data. can be tweaked?
    def objective(H):
        mu = H[0]
        B  = H[1:]
        rate = np.exp(mu+X.dot(B))
        like = K*mu+Z.dot(B)-np.sum(rate)-np.sum(penalties*B**2)
        return -like*scale
    def gradient(H):
        mu = H[0]
        B  = H[1:]
        rate  = np.exp(mu+X.dot(B))
        dmu   = K-np.sum(rate)
        dpenalty = 2*penalties*B
        dbeta = Z.T-X.T.dot(rate) - dpenalty
        grad = np.append(dmu, dbeta)
        return -grad*scale
    def hessian(H):
        mu = H[0]
        B  = H[1:]
        rate  = np.exp(mu+X.dot(B))
        dmumu = np.sum(rate)
        dmuB  = X.T.dot(rate)
        dBB   = X.T.dot(rate[:,None]*X)
        ddpen = np.diag(np.ones(len(B)))*penalties*2
        hess  = np.zeros((len(H),)*2)
        hess[0 ,0 ] = dmumu
        hess[0 ,1:] = dmuB
        hess[1:,0 ] = dmuB.T
        hess[1:,1:] = ddpen + dBB
        return hess*scale
    return objective, gradient, hessian

def ppglmfit(X,Y,verbose=False):
    '''
    The GLM solver in statsmodels is very general. It accepts any link
    function and expects that, if you want a constant term in your model,
    that you have already manually added a column of ones to your
    design matrix. This wrapper simplifies using GLM to fit the common
    case of a Poisson point-process model, where the constant term has
    not been explicitly added to the design matrix

    Parameters
    ----------
    X: N_observations x N_features design matrix.
    Y: Binary point process observations

    Returns
    -------
    μ, B: the offset and parameter estimates for the GLM model.
    '''
    # add constant value to X, if the 1st column is not constant
    if np.mean(Y)>0.1 and verbose:
        print('Caution: spike rate very high, is Poisson assumption valid?')
    if np.sum(Y)<100 and verbose:
        print('Caution: fewer than 100 spikes to fit model')
    if not all(X[:,0]==X[0,0]):
        X = np.hstack([np.ones((X.shape[0],1),dtype=X.dtype), X])
    poisson_model   = GLM(Y,X,family=Poisson())
    poisson_results = poisson_model.fit()
    M = poisson_results.params
    return M[0],M[1:]


def fitGLM(X,Y,L2Penalty=0.0,mu0=None,B0=None,**kwargs):
    '''
    Fit the model using gradient descent with hessian
    
    Parameters
    ----------
    X : matrix
        design matrix
    Y : vector
        binary spike observations
    L2Penalty : scalar
        optional L2 penalty on features, defaults to 0
    mu0 : scalar or None
        Initial guess for mean offset. 
        Optional, defaults to 0 if None.
    B0 : vector or None
        Initial guess for parameter weights. 
        Optional, defaults to zeroes if None
        
    Other Parameters
    ----------------
    kwargs : 
        Keyword arguments to forward to `scipy.optimize.minimize`

    Returns
    -------
    mu : mean offset parameter
    B : feature weights
    '''
    objective, gradient, hessian = GLMPenaltyL2(X,Y,L2Penalty)
    initial = np.zeros(X.shape[1]+1)
    if not mu0 is None:
        initial[0] = mu0
    if not B0 is None:
        initial[1:] = B0
    M = minimize(objective,initial,
        jac=gradient,hess=hessian,method='Newton-CG')['x']
    mu,B = M[0],M[1:]
    return mu,B

def crossvalidatedAUC(X,Y,NXVAL=4):
    '''
    Crossvalidated area under the ROC curve calculation. This routine
    uses the non-regularized GLMPenaltyL2 to fit a GLM point-process 
    model and test accuracy under K-fold crossvalidation.
    
    Parameters
    ----------
    X : np.array
        Covariate matrix Nsamples x Nfeatures
    Y : np.array
        Binary point-process observations, 1D array length Nsamples 
    NXVAL : positive int
        Defaults to 4. Number of cross-validation blocks to use
        
    Returns
    -------
    float
        Area under the ROC curve, cross-validated, for non-regularized
        GLM point process model fit
    '''
    N = X.shape[0]
    P = np.random.permutation(N)
    X = X[P,:]
    Y = Y[P]
    blocksize = N//NXVAL
    predicted = []
    M = np.zeros(X.shape[1]+1)
    for i in range(NXVAL):
        a = i*blocksize
        b = a + blocksize
        if i==NXVAL-1: b = N
        train_X = concatenate([X[:a,:],X[b:,:]])
        train_Y = concatenate([Y[:a],Y[b:]])
        objective, gradient, hessian = GLMPenaltyL2(train_X,train_Y,0)
        M = minimize(objective,M,jac=gradient,hess=hessian,method='Newton-CG')['x']
        mu,B = M[0],M[1:]
        predicted.append(mu + X[a:b,:].dot(B))
    return auc(Y,concatenate(predicted))

def gradientglmfit(X,Y,L2Penalty=0.0):
    '''
    mu_hat, B_hat = gradientglmfit(X,Y,L2Penalty=0.0)
    
    Fit Poisson GLM using gradient descent with hessian
    
    Parameters
    ----------
    X : np.array
        Covariate matrix Nsamples x Nfeatures
    Y : np.array
        Binary point-process observations, 1D array length Nsamples 
    '''
    objective, gradient, hessian = GLMPenaltyL2(X,Y,L2Penalty)
    initial = np.zeros(X.shape[1]+1)
    M = minimize(objective, initial,
        jac   =gradient,
        hess  =hessian,
        method='Newton-CG')['x']
    mu_hat,B_hat = M[0],M[1:]
    return mu_hat, B_hat
    
def cosine_kernel(x):
    '''
    Raised cosine basis kernel, normalized such that it integrates to 1
    centered at zero. Time is rescaled so that the kernel spans from
    -2 to 2
    
    Parameters
    ----------
    x : vector
    
    Returns
    -------
    vector
        $\\tfrac 1 4 + \\tfrac 1 2 cos(x)$ if $x\\in[-\\pi,\\pi]$, otherwise 0.
    '''
    x = np.float64(np.abs(x))/2.0*np.pi
    return np.piecewise(x,[x<=np.pi],[lambda x:(np.cos(x)+1)/4.0])

def log_cosine_basis(N=range(1,6),t=np.arange(100),base=2,offset=1,normalize=True):
    '''
    Generate overlapping log-cosine basis elements
    
    Parameters
    ----------
    N : array 
        Array of wave quarter-phases
    t : array
        times
    base : scalar
        exponent base
    offset : scalar
        leave this set to 1 (default)
    
    Returns
    -------
    B : array
        Basis with n_elements x n_times shape
    '''
    s = np.log(t+offset)/np.log(base)
    kernels = np.array([cosine_kernel(s-k) for k in N]) # evenly spaced in log-time
    if normalize:
        kernels = kernels/np.log(base)/(offset+t) # correction for change of variables, kernels integrate to 1 now
    return kernels

def make_cosine_basis(N,L,min_interval,normalize=True):
    '''
    Build N logarightmically spaced cosine basis functions
    spanning L samples, with a peak resolution of min_interval
    
    # Solve for a time basis with these constraints
    # t[0] = 0
    # t[min_interval] = 1
    # log(L)/log(b) = n_basis+1
    # log(b) = log(L)/(n_basis+1)
    # b = exp(log(L)/(n_basis+1))
    
    Parameters
    ----------
    N : int
        Number of basis functions
    L : int
        Number of time-bins
    min_interval : scalar
        Number of bins between the two shortes basis elements. That is,
        minimum time separation between basis functions.
        
    Returns
    -------
    B : array
        Basis with n_elements x n_times shape
    '''
    t = np.arange(L)/min_interval+1
    b = np.exp(np.log(t[-1])/(N+1))
    B = log_cosine_basis(np.arange(N),t,base=b,offset=0,normalize=normalize)
    return B

# Argument cleaning: tolerate some sloppiness in return types
from neurotools.linalg.arguments import scalar,asvector

def numeric_grad(obj,p,delta=np.finfo('float32').eps):
    '''
    Numerically estimate the gradient of an objective function
    '''
    p = np.float64(p).ravel()
    N = len(p)
    g = np.zeros(N)
    x = scalar(obj(p))
    for i in range(N):
        dp     = p.copy()
        dp[i] += delta
        dx     = scalar(obj(dp))
        g[i]   = (dx-x)/delta
    return g


def numeric_hess(jac,p,delta=np.finfo('float32').eps):
    '''
    Numerically estimate the hessian given a gradient function
    '''
    p = np.float64(p)
    N = len(p)
    H = np.zeros((N,N))
    g = asvector(jac(p))
    for i in range(N):
        for j in range(N):
            dp     = p.copy()
            dp[i] += delta
            dg     = asvector(jac(dp))
            H[i]   = (dg-g)/delta
    return H

#############################################################################

if __name__=='__MAIN__' or __name__=='__main__':

    import datetime
    import time as systime

    def current_milli_time():
        return int(round(systime.time() * 1000))
    
    #stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    __GLOBAL_TIC_TIME__ = None
    
    def tic(st=''):
        ''' Similar to Matlab tic '''
        global __GLOBAL_TIC_TIME__
        t = current_milli_time()
        try:
            __GLOBAL_TIC_TIME__
            if not __GLOBAL_TIC_TIME__ is None:
                print('t=%dms'%((t-__GLOBAL_TIC_TIME__)),st)
            else: print("timing...")
        except: print("timing...")
        __GLOBAL_TIC_TIME__ = current_milli_time()
        return t
        
    def toc(st=''):
        ''' Similar to Matlab toc '''
        global __GLOBAL_TIC_TIME__
        t = current_milli_time()
        try:
            __GLOBAL_TIC_TIME__
            if not __GLOBAL_TIC_TIME__ is None:
                print('dt=%dms'%((t-__GLOBAL_TIC_TIME__)),st)
            else:
                print("havn't called tic yet?")
        except: print("havn't called tic yet?")
        return t

    Fs = 1000  # Sampling rate ( Hz )
    dt = 1./Fs # bin width ( seconds )
    T  = 450.  # Time duration ( seconds )
    N  = int(T*Fs)  # number of time bins

    '''
    Simple model: K Gaussian features
       λ(y)  = exp{ μ + BX }
    ln(λ(y)) =      μ + BX
    '''
    mu = -6
    K  = 4                   # number of features
    B  = np.random.randn(K)         # draw randomly model weights
    X  = np.random.randn(N,K)       # simulate covariate data
    L  = np.exp(mu+X.dot(B)) # simulat conditional intensities
    Y  = np.random.rand(N)<L        # draw spike train from conditional intensity
    print('Simulated',T,'seconds')
    print(np.sum(Y),'spikes at a rate of',np.sum(Y)/T,'Hz')

    # split into training and validation data
    split = N//2 # The operator // is integer division in Python
    X_train,X_validate = X[:split],X[split:]
    L_train,L_validate = L[:split],L[split:]
    Y_train,Y_validate = Y[:split],Y[split:]

    # Fit the model using GLM from statsmodels
    print('\nFitting using IRLS')
    tic()
    mu_hat,B_hat = ppglmfit(X_train,Y_train)
    toc()
    L_hat = np.exp(X.dot(B_hat)+mu_hat)
    L_hat_train,L_hat_validate = L_hat[:split],L_hat[split:]
    print('The true model is   μ,B =',mu,B)
    print('GLM fit found       μ,B =',mu_hat,B_hat)
    print('AUC on true      model is',auc(Y_train,L_train))
    print('AUC on training   data is',auc(Y_train,L_hat_train))
    print('AUC on validation data is',auc(Y_validate,L_hat_validate))

    # Fit the model using gradient descent without hessian
    print('\nFitting using conjugate gradient (no Hessian)')
    tic()
    objective, gradient, hessian = GLMPenaltyL2(X_train,Y_train,0)
    M = minimize(objective, np.zeros(len(B)+1), jac=gradient)['x']
    mu_hat,B_hat = M[0],M[1:]
    toc()
    L_hat = np.exp(X.dot(B_hat)+mu_hat)
    L_hat_train,L_hat_validate = L_hat[:split],L_hat[split:]
    print('The true model is   μ,B =',mu,B)
    print('minimize found      μ,B =',mu_hat,B_hat)
    print('AUC on true      model is',auc(Y_train,L_train))
    print('AUC on training   data is',auc(Y_train,L_hat_train))
    print('AUC on validation data is',auc(Y_validate,L_hat_validate))

    # Fit the model using gradient descent with hessian
    print('\nFitting using conjugate gradient with Hessian')
    tic()
    objective, gradient, hessian = GLMPenaltyL2(X_train,Y_train,0)
    initial = np.zeros(len(B)+1)
    M = minimize(objective, initial,
        jac=gradient,
        hess=hessian,
        method='Newton-CG')['x']
    mu_hat,B_hat = M[0],M[1:]
    toc()
    L_hat = np.exp(X.dot(B_hat)+mu_hat)
    L_hat_train,L_hat_validate = L_hat[:split],L_hat[split:]
    print('The true model is   μ,B =',mu,B)
    print('minimize found      μ,B =',mu_hat,B_hat)
    print('AUC on true      model is',auc(Y_train,L_train))
    print('AUC on training   data is',auc(Y_train,L_hat_train))
    print('AUC on validation data is',auc(Y_validate,L_hat_validate))

    print('\nSanity checking L2 penalized gradient and hessian code')
    # Sanity check for the L2 gradient: confirm that the numeric gradient
    # and hessian agree with that returned by the gradient and hessian
    # functions.
    # just printing one significant figure for now, to reduce clutter
    np.set_printoptions(precision=1)
    mu = -4
    B  = np.random.randn(5)
    X  = np.random.randn(N,5)
    L  = np.exp(mu+X.dot(B))
    Y  = np.random.rand(N)<L
    objective, gradient, hessian = GLMPenaltyL2(X,Y,10)
    p = np.random.randn(len(B)+1)
    delta = 0.01
    numeric_gradient = np.zeros(len(p))
    for i in range(len(p)):
        q = np.array(p)
        q[i] += delta
        numeric_gradient[i] = (objective(q)-objective(p))/delta
    numeric_hessian = np.zeros((len(p),)*2)
    for i in range(len(p)):
        dp = gradient(p)
        for j in range(len(p)):
            q = np.array(p)
            q[j] += delta
            dq = gradient(q)
            numeric_hessian[i,j]=(dq-dp)[i]/delta
    print('negative log likelihood at',p,'is',objective(p))
    print('gradient of neg loglike at',p,'is',gradient(p))
    print('numeric   Δ of -loglike at',p,'is',numeric_gradient)
    print('hessian  of neg loglike at',p,'is\n',hessian(p))
    print('numeric  Δ² of -loglike at',p,'is\n',numeric_hessian)


    
    
def __history_basis(K=5, tstop=100, scale=10, normalize=0):
    '''
    Generate raised cosine history basis. 
    
    TODO: this function is duplicated in several locations, 
    consolidate.
    
    >>> basis = history_basis(4,100,10)
    >>> plot(basis.T)
    
    Parameters
    ----------
    K : int
        Number of basis elements. Defaults to 5
    tstop : int
        Time-point at which to stop. Defaults to 100
    scale : int
        Exponent basis for logarithmic time rescaline. Defaults to 10
    
    '''
    if not normalize==0:
        raise NotImplementedError('Normalization options have not been implemented yet');
    time    = arange(tstop)+1
    logtime = log(time+scale)
    a,b     = np.min(logtime),np.max(logtime)
    phases  = (logtime-a)*(1+K)/(b-a)
    return array([0.5-0.5*cos(clip(phases-i+2,0,4)*pi/2) for i in range(K)])

    
    
    
    
