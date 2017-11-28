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

def history_basis(K=5, tstop=100, scale=10, normalize=0):
    '''
    Generate raised cosine history basis. 
    TODO: this function is duplicated in several locations, consolidate.
    
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


