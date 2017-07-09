#!/usr/bin/python
# -*- coding: UTF-8 -*-
# BEGIN PYTHON 2/3 COMPATIBILITY BOILERPLATE
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

'''
Very simple demonstration of driving damped oscillators
'''

def resonantDrive(s,f,tau,x,Fs=1000):
    '''
    Drive resonator with frequency f and time constant tau and state s
    with signal state x
    f   in Hz
    tau in seconds
    Fs  in samples/second
    
    Convert tau into discrete update:
    x(t)  = exp(-t/tau)
    dx/dt = -1/tau x(t)
    tau dx/dt = -x(t)
    x = x + dt*(-1/tau x(t))
    x = x - dt/tau x
    x = (1 - dt/tau) x
    
    Convert frequency into radians
    f  is in cycles  / second
    Fs is in samples / second
    We need radians  / sample
    There are 2 pi radians / cycle
    theta = f*2*pi/Fs
    '''
    theta = f*2*pi/Fs
    s *= exp(1j*theta)
    dt = 1./Fs
    alpha = (1-dt/tau)
    s  = alpha*s+(1-alpha)*x
    return s

def resonantFilter(x,f,tau,Fs=1000):
    '''
    need to normalize
    integrate
    x(t)  = exp(-t/tau)
    from 0 to iy
    skip for now
    '''
    N = len(x)
    M = len(f)
    print(N,M)
    assert N==M
    dt = 1./Fs
    s = complex(0)
    result = complex64(zeros(N))
    adjustor = 1.0
    a  = dt/tau
    z  = 2j*pi/Fs
    x *= a
    b  = 1-a
    modulator = b*exp(f*z)
    for i in xrange(N):
        s = s*modulator[i]+x[i]
        result[i]=s
    return result

def resonantFiltfilt(x,f,tau,order=1):
    x = padout(x)
    f = padout(f)
    for i in range(order):
        x = resonantFilter(x,f,tau)
        x = resonantFilter(x[::-1],-f,tau)[::-1]
    return padin(x)

def resonantFilter(x,f,tau,Fs=1000):
    N = len(x)
    M = len(f)
    print(N,M)
    assert N==M
    dt = 1./Fs
    s = complex(x[0])
    result = complex64(zeros(N))
    adjustor = 1.0
    a  = dt/tau
    z  = 2j*pi/Fs
    b  = 1-a
    modulator = b*exp(f*z)
    for i in xrange(N):
        p = s*modulator[i]
        r = sin(angle(p))*p*x[i]/(0.01+real(p)) + cos(angle(p))*(x[i]-real(p))
        s = p + a*r
        result[i]=s
    return result








    
