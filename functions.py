#!/usr/bin/python
# -*- coding: UTF-8 -*-
# BEGIN PYTHON 2/3 COMPATIBILITY BOILERPLATE
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

"""
Commonly used functions
"""

import numpy as np

# Constants: ensure compatibility with float32
# while using highest available accuracy (longdouble)

F32EPS     = np.longdouble('7e-45')
F32SAFE    = np.sqrt(F32EPS)
F64EPS     = np.longdouble('1.4012985e-45')
F64SAFE    = np.sqrt(F64EPS)
ZERO128    = np.longdouble('0')
EMAX       = np.longdouble(np.sqrt(np.log(np.finfo(np.float64).max)))
F128EMAX   = np.sqrt(np.longdouble('11355.52340629414395'))

lgE        = np.longdouble('1.442695040888963407359924681001892137426645954152985934135')
pi         = np.longdouble('3.141592653589793238462643383279502884197169399375105820974')
tau        = np.longdouble('6.283185307179586476925286766559005768394338798750211641949')
e          = np.longdouble('2.718281828459045235360287471352662497757247093699959574966')
sqrt2      = np.longdouble('1.414213562373095048801688724209698078569671875376948073176')
sqrttau    = np.longdouble('2.506628274631000502415765284811045253006986740609938316629')
invsqrttau = np.longdouble('0.398942280401432677939946059934381868475858631164934657666')

# largest floating point accuracy that scipy.linalg
# can support
LINALGMAXFLOAT = np.float64

def slog(x,eps=F64SAFE,returntype=LINALGMAXFLOAT):
    '''
    "safe" natural logarithm function, clips values avoiding NaN and inf
    '''
    return returntype(np.log(np.maximum(eps,x)))

def sexp(x,limit=EMAX,returntype=LINALGMAXFLOAT):
    '''
    "safe" exponential function, clips values avoiding NaN and inf
    '''
    limit = np.longdouble(limit)
    x = np.longdouble(x)
    x = np.clip(x,-limit,limit)
    return returntype(np.exp(x))

def sigmoid(x,limit=EMAX,returntype=LINALGMAXFLOAT):
    '''
    sigmoid function 1/(1+exp(-x))
    '''
    # logaddexp(x1,x2) = log(exp(x1) + exp(x2))
    limit = np.longdouble(limit)
    x = np.longdouble(x)
    x = np.clip(x,-limit,limit)
    return returntype(sexp(-np.logaddexp(ZERO128,-np.longdouble(x))))

def inversesigmoid(x,returntype=LINALGMAXFLOAT):
    '''
    Inverse of sigmoid function 1/(1+exp(-x)), -[log(1-x)+log(x)]
    '''
    return returntype(slog(x)-slog(1-x))

def dsigmoid(x,returntype=LINALGMAXFLOAT): 
    '''
    Fist derivative of sigmoid
    '''
    x = np.longdouble(x)
    return sexp(\
        -np.logaddexp(ZERO128,-x)\
        -np.logaddexp(ZERO128,x),
        returntype=returntype)

# Sigmoid and derivatives

def g(x,returntype=LINALGMAXFLOAT): 
    '''
    Evaluates g(x)=log(1+exp(x)) as accurately as possible. 
    '''
    return returntype(np.logaddexp(ZERO128,np.longdouble(x)))
    
def f(x,returntype=LINALGMAXFLOAT): 
    '''
    evaluates f(x)=1/(1+exp(-x)) as accurately as possible
    '''
    return returntype(sexp(-np.logaddexp(ZERO128,-np.longdouble(x))))

def f1(x,returntype=LINALGMAXFLOAT): 
    '''
    Fist derivative of sigmoid
    '''
    x = np.longdouble(x)
    return sexp(\
        -np.logaddexp(ZERO128,-x)\
        -np.logaddexp(ZERO128,x),
        returntype=returntype)
    
def f2(x,returntype=LINALGMAXFLOAT):
    '''
    Second derivative of sigmoid
    
    (q - p) p q
    '''
    x = np.longdouble(x)
    logp = -np.logaddexp(ZERO128,-x)
    logq = -np.logaddexp(ZERO128, x)
    p  = np.exp(np.minimum(F128EMAX,logp))
    q  = np.exp(np.minimum(F128EMAX,logq))
    return returntype((q-p)*q*p);

def npdf(mu,sigma,x):
    '''
    Univariate Gaussian probability density
    
    Parameters
    ----------
    mu : float, scalar or array-like 
        Mean(s) of distribution(s)
    sigma : float, scalar or array-like 
        Standard deviation(s) of distribution(s)
    x : float, scalar or array-like 
        Points at which to evaluate distribution(s)
    '''
    mu    = np.array(mu).ravel()
    sigma = np.array(sigma).ravel()
    x     = np.array(x).ravel()
    invsigma = 1.0/sigma
    x = (x-mu)*invsigma
    return (invsqrttau*invsigma) * sexp(-0.5*x**2)

def log_factorial(k):
    '''
    Returns the logarithm of a factorial by taking the sum of the
    logarithms of 1..N. Slow, but numerically more accurate than
    taking the logarithm of the factorial or using approximations.
    
    k should be an integer.
    '''
    return 1 if k<2 else np.sum([np.log(i) for i in range(1,k+1)])


