#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Kalman filtering impementation for demonstration
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

from neurotools.nlab import *

from pylab import *
from numpy import *


def multiply_gaussian(M1,C1,M2,C2):
    C1P2   = ldiv(C2,C1).T
    I      = np.eye(C2.shape[0])
    IC1P2  = I+C1P2
    m      = ldiv(IC1P2,M1 + C1P2@M2)
    c      = ldiv(IC1P2,C1)
    return m,c

def kalman_forward(m,c,A,Q):
    m   = A@m
    c   = A@c@A.T + Q
    return m,c

def kalman_backward(m,c,A,Q):
    m   = ldiv(A,m)
    c   = ldiv(A,ldiv(A,c).T) + Q 
    return m,c

def kalman_measure(m,c,B,pxyBi,y):
    # Measure
    cpxyBi = c@pxyBi
    I      = np.eye(m.shape[0])
    Icpxy  = I+cpxyBi@B
    m      = ldiv(Icpxy,m + cpxyBi@y)
    c      = ldiv(Icpxy,c)
    return m,c

def kalman_smooth(Y,A,B,Q,U):
    # initial mean and covariance
    N = Q.shape[0]
    K = U.shape[0]
    pxyBi = ldiv(U,B).T
    m     = zeros(N)
    c     = np.eye(N)*10 
    Mf,Cf = [m],[c]
    for i,y in enumerate(Y):
        m,c = kalman_forward(m,c,A,Q)
        m,c = kalman_measure(m,c,B,pxyBi,y)
        Mf +=[m.copy()]
        Cf +=[c.copy()]
    Mf,Cf = array(Mf),array(Cf)
    Mb,Cb = [m],[c]
    for i,y in enumerate(Y[::-1,:]):
        m,c = kalman_backward(m,c,A,Q)
        Mb +=[m.copy()]
        Cb +=[c.copy()]
        m,c = kalman_measure(m,c,B,pxyBi,y)
    Mb,Cb = array(Mb)[::-1,:],array(Cb)[::-1,:]
    # Combine forward/backward
    Mp,Cp = [],[]
    for M1,C1,M2,C2 in list(zip(Mf,Cf,Mb,Cb))[1:]:
        mp,cp = multiply_gaussian(M1,C1,M2,C2)
        Mp.append(mp)
        Cp.append(cp)
    return array(Mp).real,array(Cp).real

