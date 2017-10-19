#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
import sys
# more py2/3 compat
from neurotools.system import *

'''
Functions for computing the log-PDF of common distributions.
These yield a more digits of precision than their counterparts in
scipy.stats by computing log-probability values using high precision
128-bit floats.
'''

import numpy as np
from neurotools.functions import log_factorial, slog
import random
import scipy
import scipy.special

def poisson_logpdf(k,l):
    '''
    Gives the log-pdf for a poisson distribution with rate l 
    evaluated at points k. k should be a vector of integers.
    '''
    # k,l = map(np.float128,(k,l))
    return k*slog(l)-l-np.array([scipy.special.gammaln(x+1) for x in k])
    #return k*slog(l)-l-np.array([log_factorial(x) for x in k])

def poisson_pdf(k,l):
    return np.exp(poisson_logpdf(k,l))

# log(sqrt(2*pi)) computed to high precision on Wolfram Alpha.
logsqrt2pi = np.float128(
    '0.91893853320467274178032973640561763986139747363778341281')

def gaussian_logpdf(mu,sigma,x):
    '''
    Non-positive standar deviations will be clipped
    '''
    mu,sigma,x = map(np.float128,(mu,sigma,x))
    x = (x-mu)/sigma
    return -0.5*x*x - slog(sigma) - logsqrt2pi

def gaussian_pdf(mu,sigma,x):
    return np.exp(gaussian_logpdf(mu,sigma,x))
    
def explogpdf(x,dx=1):
    '''
    Convert log-pdf to normalized pdf, integrating to get normalization constant
    '''
    x -= np.mean(x)
    p = np.exp(x)
    return p/(sum(p)*dx)
    
def sample_categorical(pobabilities):
    '''
    Pick a state according to probabilities

    Parameters
    ----------
    probabilities : vector 
        Vector of probabilities, must sum to 1.
    
    Returns
    -------
    i : int
        integer between 0 and len(probabilities)-1
    '''
    pobabilities = np.ravel(pobabilities)
    r = random.uniform(0,np.sum(pobabilities))
    cumulative = 0.
    for i,pr in enumerate(pobabilities):
        if cumulative+pr>=r: return i
        cumulative += pr
    assert False
