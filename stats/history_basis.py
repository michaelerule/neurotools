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
    K: number of basis elements
    tstop: length of time to cover
    scale: affects how "logarithmic" basis scaling looks
    normalize: normalize bases to unit or not?

    >>> basis = history_basis(4,100,10)
    >>> plot(basis.T)
    normalization not implemented
    '''
    assert normalize==0
    time    = arange(tstop)+1
    logtime = log(time+scale)
    a,b     = np.min(logtime),np.max(logtime)
    phases  = (logtime-a)*(1+K)/(b-a)
    return array([0.5-0.5*cos(clip(phases-i+2,0,4)*pi/2) for i in range(K)])


