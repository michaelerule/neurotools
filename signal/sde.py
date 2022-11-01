#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Functions to sample Wiener/OU process.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np

def sample_wiener_process(x0,sigma,dt,N,ntrial=1):
    '''
    Sample from the standard Wiener process. 
    
    Parameters
    ----------
    x0: float
        Initial conditions
    sigma: positive float
        Standard deviation of driving Wiener process
    dt: positive float
        Time step
    N: positive int
        Number of samples to draw 
    
    Returns
    -------
    '''
    simulated = np.zeros((N,ntrial),dtype=np.float32)
    x = x0*np.ones((ntrial,),dtype=np.float32)
    for i in range(N):
        x += sigma * np.random.randn(ntrial) * np.sqrt(dt)
        simulated[i] = x
    return simulated

def sample_ou_process(x0,sigma,tau,dt,N,ntrial=1):
    '''
    Sample from an Ornstein-Uhlenbeck process.
    
    Parameters
    ----------
    x0: float
        Initial conditions
    sigma: positive float
        Standard deviation of driving Wiener process
    tau: positive float
        Exponential damping time constant
    dt: positive float
        Time step
    N: positive int
        Number of samples to draw 
    
    Returns
    -------
    simulated:
    '''
    simulated = np.zeros((N,ntrial),'float')
    x = x0*np.ones((ntrial,),'float')
    for i in range(N):
        x += -(1./tau)*x*dt + sigma * np.random.randn(ntrial) * np.sqrt(dt)
        simulated[i] = x
    return simulated
