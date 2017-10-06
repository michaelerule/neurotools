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
