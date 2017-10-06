#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
# more py2/3 compat
from neurotools.system import *

'''
# Izhikevich model
'''

import numpy as np

def dv_izh(u,v,I):
    '''
    Time derivative for $v$ variable in Izhikevich model
    '''
    return (0.04*v+5.0)*v+140.0-u+I

def du_izh(u,v,a,b):
    '''
    Time derivative for $u$ variable in Izhikevich model
    '''
    return a*(b*v-u)

def update_izh(u,v,a,b,c,d,I,dt=1):
    '''
    Izhikevich neuron state update
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
    '''
    L = len(signal)
    u = 0
    v = -0
    state = []
    for i in range(L):
        u,v,y = update_izh(u,v,a,b,c,d,signal[i],dt)
        state.append(np.array([u,v,y]))
    return np.array(state)
