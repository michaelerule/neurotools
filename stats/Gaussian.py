#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

import numpy as np
from neurotools.stats.distributions import poisson_logpdf
import random
from scipy.optimize import minimize
from neurotools.functions import slog,sexp

def gaussian_quadrature(p,domain,eps=1e-12):
    '''
    Numeric integration of probability density `p` over `domain`; Values
    in `p` smaller than `eps` (defaults to 1e-12) are set to `eps`.
    
    Treat f as a density and estimate it's mean and precision
    over the domain
    
    Parameters
    ----------
    p
        1D iterable of probabilities
    domain
        1D iterable of domain points corresponding to `p`
    eps=1e-12
        Minimum probability value permitted
    
    Returns
    -------
    Gaussian
        Gaussian object with mean and precision matching estimated mean
        and precision of distribution specified by (`p`,`domain`).
    '''
    assert np.all(np.isfinite(p))
    p = np.array(p)
    p[p<eps] = eps   
    p/= np.sum(p)
    m = np.sum(domain*p)
    assert np.isfinite(m)
    v = np.sum((domain-m)**2*p)
    assert np.isfinite(v)
    t = 1./(v+1e-10)
    assert np.isfinite(t)
    assert t>=0
    return Gaussian(m, t)

def gaussian_quadrature_logarithmic(logp,domain):
    '''
    Numeric integration of log-probability. Not yet implemented.
    
    Parameters
    ----------
    p
        1D iterable of probabilities
    domain
        1D iterable of domain points corresponding to `p`
    eps=1e-12
        Minimum probability value permitted
    
    Returns
    -------
    
    '''
    raise NotImplementedError('This function is not yet implemented')
    normalization = sum(p)
    m = np.sum(domain*p)/normalization
    assert np.isfinite(m)
    v = np.sum((domain-m)**2*p/normalization)
    assert np.isfinite(v)
    t = 1./(v+1e-10)
    assert np.isfinite(t)
    assert t>=0
    return Gaussian(m, t)

class Gaussian:
    '''
    Gaussian model to use in abstracted forward-backward
    Supports multiplication of Gaussians

    Parameters
    ----------
    m: mean 
    t: precision (reciprocal of variance)
    
    Returns
    -------
    '''
    def __init__(s,m,t):
        '''
    
        Parameters
        ----------
        m : 
            Mean of distribution
        t : 
            Precision (1/variance) of distribution
        
        Returns
        -------
        Gaussian
            Gaussian density object
        '''
        s.m,s.t = m,t
        s.lognorm = 0
    def __mul__(s,o):
        '''
        Multiply two Gaussians together in terms of probability, for example
        to apply Bayes rule. 
    
        Parameters
        ----------
        o : Gaussian
            Another Gaussian. Can also be 1 to indicate the identity
        
        Returns
        -------
        Gaussian 
            New gaussian distribution that is the product of this 
            distribution with the distribution `o`
        '''
        if o==1: return s
        assert isinstance(o,Gaussian)
        t = s.t+o.t
        m = (s.m*s.t + o.m*o.t)/(t) if abs(t)>1e-16 else s.m+o.m        
        assert np.isfinite(t)
        assert np.isfinite(m)
        result = Gaussian(m,t)
        # propagating normalization factors for forward-backward
        if hasattr(o,'lognorm'):
            result.lognorm = s.lognorm+o.lognorm
        else:
            result.lognorm = s.lognorm
        return result
    def __truediv__(s,o):
        '''
        Divide this Gaussian distribtion by another Gaussian distribution
        `o`
    
        Parameters
        ----------
        o : Gaussian
            Another Gaussian. Can also be 1 to indicate the identity
        
        Returns
        -------
        Gaussian 
            New gaussian distribution reflecting division of this Gaussian
            distribution by the distribution `o`
        '''
        if o==1: return s
        assert isinstance(o,Gaussian)
        t = s.t-o.t
        m = (s.m*s.t - o.m*o.t)/(t) if abs(t)>1e-16 else s.m-o.m
        assert np.isfinite(t)
        assert np.isfinite(m)
        result = Gaussian(m,t)
        # propagating normalization factors for forward-backward
        if hasattr(o,'lognorm'):
            result.lognorm = s.lognorm-o.lognorm
        else:
            result.lognorm = s.lognorm
        return result
    __div__  = __truediv__
    def __call__(s,x): 
        '''
        Evaluate this Gaussian PDF at location `x`
    
        Parameters
        ----------
        x : float or np.array
            Point or array of points at which to evaluate the PDF
        
        Returns
        -------
        np.array 
            PDF value at locations specified by `x`
        '''
        return sexp(-0.5*s.t*(x-s.m)**2)*np.sqrt(s.t/(2*np.pi))
    def __str__(s):
        return 'm=%0.4f, t=%0.4f'%(s.m,s.t)
    def logpdf(s,x):
        '''
        The log-pdf of a univariate Gaussian
    
        Parameters
        ----------
        x : float or np.array
            Point or array of points at which to evaluate the log-PDF
        
        Returns
        -------
        np.array 
            log-PDF value at locations specified by `x`
        '''
        return -0.5*s.t*(x-s.m)**2 + 0.5*slog(s.t)-0.91893853320467267#-0.5*log(2*np.pi)
    
