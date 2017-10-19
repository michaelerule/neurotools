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
from neurotools.functions import slog
from neurotools.stats.distributions import poisson_logpdf
import random
from scipy.optimize import minimize

def gaussian_quadrature(p,domain):
    '''
    Treat f as a density and estimate it's mean and precision
    over the domain
    '''
    p/= np.sum(p)
    m = np.sum(domain*p)
    assert np.isfinite(m)
    v = np.sum((domain-m)**2*p)
    assert np.isfinite(v)
    t = 1./(v+1e-10)
    assert np.isfinite(t)
    assert t>=0
    return Gaussian(m, t)

class Gaussian:
    '''
    Gaussian model to use in abstracted forward-backward
    Supports multiplication of Gaussians

    m: mean 
    t: precision (reciprocal of variance)
    '''
    def __init__(s,m,t):
        s.m,s.t = m,t
        s.lognorm = 0
    def __mul__(s,o):
        if o is 1: return s
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
        if o is 1: return s
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
    __call__ = lambda s,x: np.exp(-0.5*s.t*(x-s.m)**2)*np.sqrt(s.t/(2*np.pi))
    __str__  = lambda s: 'm=%0.4f, t=%0.4f'%(s.m,s.t)
    def logpdf(s,x):
        '''
        The log-pdf of a univariate Gaussian
        '''
        return -0.5*s.t*(x-s.m)**2 + 0.5*log(s.t)-0.91893853320467267#-0.5*log(2*np.pi)
    
