#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Functions for computing the log-PDF of common distributions.

Some of these yield a more digits of precision than their 
counterparts in `scipy.stats` by computing log-probability 
values using `np.longdouble`.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import random
import scipy
import scipy.special
from neurotools.util.functions import log_factorial, slog, sexp

# log(sqrt(2*pi)) computed to high precision
logsqrt2pi = np.longdouble('0.91893853320467274178032973640561763986139747363778341281')

def poisson_logpdf(k,l):
    '''
    Evaluate the log-pdf for a poisson distribution with 
    rate `l` evaluated at points k.
    
    Parameters
    ----------
    k: np.int32
        Counts at whih to evaluate the log-pdf
    l: positive float
         Poisson rate
    
    Returns
    -------
    result: np.longdouble
        Log-probability of each `k` for a Poisson 
        distribution with rate `l`.
    '''
    return k*slog(l)-l-np.array([scipy.special.gammaln(x+1) for x in k])

def poisson_pdf(k,l):
    '''
    Evaluate the pdf for a poisson distribution with rate 
    `l` evaluated at points k.
    
    Parameters
    ----------
    k: np.int32
        Counts at whih to evaluate the log-pdf
    l: positive float
         Poisson rate
    
    Returns
    -------
    result: np.longdouble
        Probability of each `k` for a Poisson 
        distribution with rate `l`.
    '''
    return sexp(poisson_logpdf(k,l))


def gaussian_logpdf(mu,sigma,x):
    '''
    Evaluate the log-pdf of a `(mu,sigma)` normal 
    distribution at points `x`.
    
    Parameters
    ----------
    mu: float
        Distribution mean
    sigma: positive float
        Distribution standard deviation
    x: np.float32
        Points at which to evaluate.
    
    Returns
    -------
    result: np.longdouble
        log-PDF evaluated ast `x`.
    '''
    mu,sigma,x = map(np.longdouble,(mu,sigma,x))
    x = (x-mu)/sigma
    return -0.5*x*x - slog(sigma) - logsqrt2pi

def gaussian_pdf(mu,sigma,x):
    '''
    Evaluate the pdf of a `(mu,sigma)` normal 
    distribution at points `x`.
    
    Parameters
    ----------
    mu: float
        Distribution mean
    sigma: positive float
        Distribution standard deviation
    x: np.float32
        Points at which to evaluate.
    
    Returns
    -------
    result: np.longdouble
        PDF evaluated ast `x`.
    '''
    return sexp(gaussian_logpdf(mu,sigma,x))
    
def explogpdf(x,dx=1):
    '''
    Convert log-pdf to normalized pdf, integrating to get normalization constant
    
    Parameters
    ----------
    
    Returns
    -------
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
