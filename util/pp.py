#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from matplotlib.pyplot import plot

def detect_sign_change_2D(D,k=6):
    D = np.sign(D)
    return np.float32(np.abs(D[k:,k:]+D[k:,:-k]+D[:-k,k:]+D[:-k,:-k])!=4)
    
def interpolate_zero(x,y):
    '''
    Find x such that y=f(x)=0 assuming f(x) is monotonic and crosses zero
    '''
    if not (np.any(y>0) and np.any(y<0)): return np.NaN
    o = np.argsort(y)
    y = y[o]
    x = x[o]
    return np.interp([0],y,x)[0]
    
def kill_zeros(x,eps=1e-6):
    x = np.copy(x)
    s = np.abs(x)<eps
    s[1: ] &= s[:-1]
    s[:-1] &= s[1: ]
    x[s] = np.NaN
    return x
    
def eigvalplot(f,e,i2f=lambda x:x):
    R = np.real(e)
    I = np.imag(e)
    plot(f,i2f(np.max(R,axis=1)),color='m')
    plot(f,i2f(np.min(R,axis=1)),color='m')
    plot(f,i2f(kill_zeros(np.max(I,axis=1))),color='c')
    plot(f,i2f(kill_zeros(np.min(I,axis=1))),color='c')
