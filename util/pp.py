#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pylab
from matplotlib.pyplot import plot
from scipy.optimize import root_scalar

def detect_sign_change_2D(D,k=6):
    D = np.sign(D)
    return np.float32(np.abs(D[k:,k:]+D[k:,:-k]+D[:-k,k:]+D[:-k,:-k])!=4)
    
def orderas(order,*args):
    o = np.argsort(order)
    return tuple(a[o] for a in args)
    
def interpolate_zero(x,y):
    '''
    Find x such that y=f(x)=0 assuming f(x) is monotonic and crosses zero
    '''
    x,y = orderas(x,x,y)
    return tuple(np.interp([0],y[i:i+1],x[i:i+1])[0] for i in np.where(np.diff(np.sign(y))!=0)[0])
    
def refine_zero(f,x):# Refine
    y = f(x)
    z = interpolate_zero(x,y)
    e = np.array((z[0]-1,) + z + (z[-1]+1,))
    e = (e[1:]+e[:-1])/2
    return tuple(root_scalar(f, bracket=[a,b]).root for a,b in zip(e[:-1], e[1:]))
    
def kill_zeros(x,eps=1e-12):
    x = np.float32(np.copy(x))
    s = np.abs(x)>=eps
    s[1: ] |= s[:-1]
    s[:-1] |= s[1: ]
    x[~s] = np.nan
    return x
    
def eigvalplot(f,e,i2f=lambda x:x):
    R = np.real(e)
    I = np.imag(e)
    plot(f,i2f(np.max(R,axis=1)),color='m')
    plot(f,i2f(np.min(R,axis=1)),color='m')
    plot(f,i2f(kill_zeros(np.max(I,axis=1))),color='c')
    plot(f,i2f(kill_zeros(np.min(I,axis=1))),color='c')

def signchanged2D(x):
    sg = np.sign(x)
    ssum = (\
        sg[0:-1,0:-1]+\
        sg[1:  ,0:-1]+\
        sg[0:-1,1:  ]+\
        sg[1:  ,1:  ])/4
    return np.abs(ssum)<1
    
def signchanged3D(x):
    x = np.sign(x)
    ssum = (\
        x[0:-1,0:-1,0:-1]+\
        x[1:  ,0:-1,0:-1]+\
        x[0:-1,1:  ,0:-1]+\
        x[1:  ,1:  ,0:-1]+\
        x[0:-1,0:-1,1:  ]+\
        x[1:  ,0:-1,1:  ]+\
        x[0:-1,1:  ,1:  ]+\
        x[1:  ,1:  ,1:  ])
    return np.abs(ssum)<8
    
def plot_complex_roots(
    νν,R,
    rcolor='m',
    icolor='c',
    scatter=False,
    largest_stable_oscillatory_pole=False,
    **kw):
    '''
    Parameters
    ----------
    νν np.ndarray:
        Shape Nsamples 1D array of sorted frequencies
    R np.ndarray: 
        Nroots x Nsamples array of complex roots
    '''
    if scatter:
        kw = dict(lw=0,marker='.',s=4) | kw
        plotfunction = pylab.scatter
    else:
        kw = dict(lw=1) | kw
        plotfunction = pylab.plot
    if largest_stable_oscillatory_pole:
        R[R.real>0]=-np.inf
        R[np.abs(R.imag)<1e-12]=-np.inf
        R = R[np.argmax(R.real,axis=0), np.arange(R.shape[1])][None,:]
    
    ir = R.imag
    ir.sort(axis=0)
    for r in ir:
        plotfunction(νν, kill_zeros(r),color=icolor,**kw)
        if largest_stable_oscillatory_pole:
            plotfunction(νν,-kill_zeros(r),color=icolor,**kw)
    
    rr = R.real
    rr.sort(axis=0)
    for r in rr:
        plotfunction(νν, kill_zeros(np.real(r)),color=rcolor,**kw)

