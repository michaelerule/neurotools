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


'''
Routines concerning information
'''

import numpy as np

def DKL_discrete(P,Q,eps=1e-9):
    '''
    Compute KL divergence between discrete distributions
    
    Parameters
    ----------
    P : np.array
        Vector of probabilities
    Q : np.array
        Vector of probabilities
    
    Returns
    -------
    DKL : float
        KL divergence from P to Q
    '''
    P = np.float64(P)
    Q = np.float64(Q)
    if P.shape!=Q.shape:
        raise ValueError('Arrays P and Q must be the same shape')
    P  = np.maximum(eps,P)
    P /= np.sum(P)
    Q  = np.maximum(eps,Q)
    Q /= np.sum(Q)
    return P*log(P) - P*log(Q)
