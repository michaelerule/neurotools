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
Functions related to SDEs
Experimental / under construction
'''

import numpy as np

# Sample from OU
def sample_ou_process(x0,sigma,tau,dt,N,ntrial=1):
    '''
    Parameters
    ---------
    x0    : 
        initial conditions
    sigma : 
        standard deviation of driving Wiener process
    tau   : 
        exponential damping time constant
    dt    : 
        time step
    N     : 
        number of samples to draw
    '''
    simulated = np.zeros((N,ntrial),'float')
    x = x0*np.ones((ntrial,),'float')
    for i in range(N):
        x += -(1./tau)*x*dt + sigma * np.random.randn(ntrial) * np.sqrt(dt)
        simulated[i] = x
    return simulated
