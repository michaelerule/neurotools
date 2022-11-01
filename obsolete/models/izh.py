#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Izhikevich model
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function


import numpy as np

def dv_izh(u,v,I):
    '''
    Time derivative for $v$ variable in Izhikevich model
    
    Parameters
    ----------
    u : float
        current state of u variable
    v : float
        current state of v variable
    I : float
        applied current
        
    Returns
    -------
    dv : float
        dv/dt 
    '''
    return (0.04*v+5.0)*v+140.0-u+I

def du_izh(u,v,a,b):
    '''
    Time derivative for $u$ variable in Izhikevich model
    
    Parameters
    ----------
    u : float
        current state of u variable
    v : float
        current state of v variable
    a : float
        `a` parameter from Izhikevich model
    b : float
        `b` parameter from Izhikevich model
        
    Returns
    -------
    du : float
        du/dt 
    '''
    return a*(b*v-u)

def update_izh(u,v,a,b,c,d,I,dt=1):
    '''
    Izhikevich neuron state update
    
    Parameters
    ----------
    u : float
        current state of u variable
    v : float
        current state of v variable
    a : float
        `a` parameter from Izhikevich model
    b : float
        `b` parameter from Izhikevich model
    c : float
        `c` parameter from Izhikevich model
    d : float
        `d` parameter from Izhikevich model
    I : float
        applied current
        
    Other Parameters
    ----------------
    dt : float, default 1.0
        Time step
        
    Returns
    -------
    u : float
        Updated `u` variable
    v : float
        Updated `v` variable
    y : float
        If a spike occurs, y will be a unit-volume probability mass
        i.e. 1.0/dt
    '''
    v,u = v+dt*dv_izh(u,v,I), u+dt*du_izh(u,v,a,b)
    y = 0
    if v>30:
        v = c
        u += d
        y = 1.0/dt
    return u,v,y

def sim_izh(a,b,c,d,signal,dt=1):
    '''
    Simulate response of Izhikevich neuron model to signal
    
    Parameters
    ----------
    a : float
        `a` parameter from Izhikevich model
    b : float
        `b` parameter from Izhikevich model
    c : float
        `c` parameter from Izhikevich model
    d : float
        `d` parameter from Izhikevich model
    signal : np.array
        applied current over time
        
    Other Parameters
    ----------------
    dt : float, default 1.0
        Time step
        
    Returns
    -------
    state : array
        Ntimes x 3 array of model state. 
        First column is `u` variable
        Second column is `v` variable
        Third column is spiking density
    '''
    L = len(signal)
    u = 0
    v = -0
    state = []
    for i in range(L):
        u,v,y = update_izh(u,v,a,b,c,d,signal[i],dt)
        state.append(np.array([u,v,y]))
    return np.array(state)
