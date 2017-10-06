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
import sys
if sys.version_info<(3,):
    from itertools import imap as map
# END PYTHON 2/3 COMPATIBILITY BOILERPLATEion

"""
Commonly used functions
"""

import numpy as np

def softhresh(x):
    '''
    Soft-threshold function (rescaled hyperbolic tangent)
    '''
    return 1./(1+np.exp(-x))

zero128 = np.float128('0')
def sigmoid(x):
    '''
    More numerically stable version of the logit function
    '''
    x = np.float128(x)
    return np.exp(-np.logaddexp(zero128, -x))

def npdf(mu,sigma,x):
    '''
    Gaussian probability density
    '''
    partition = 1./(sigma*np.sqrt(2*np.pi))
    x = (x-mu)/sigma
    return partition * np.exp(-0.5*x**2)

def log_factorial(k):
    '''
    Returns the logarithm of a factorial by taking the sum of the
    logarithms of 1..N. Slow, but numerically more accurate than
    taking the logarithm of the factorial or using approximations.
    
    k should be an integer.
    '''
    return 1 if k<2 else np.sum([np.log(i) for i in range(1,k+1)])

def slog(x,eps=1e-6):
    '''
    Safe natural logarithm
    Natural logarithm that truncats inputs to small positive value
    epsilon (eps, default 1e-6) to avoid underflow in the output.
    '''
    return np.log(np.maximum(eps,x))
    

emax = np.sqrt(np.log(np.float32(3.402823e38)))
def safeexp(x):
    '''
    Safe exponential
    '''
    return np.exp(np.minimum(emax,x))

