#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from pylab import *
import numpy as np
from numpy.linalg import *

def laplaceop(N):
    precision = zeros((N,))
    if (N%2==1):
        precision[N//2]=2
        precision[N//2+1]=precision[N//2-1]=-1
    else:
        precision[N//2+1]=precision[N//2]=1
        precision[N//2+2]=precision[N//2-1]=-1
    x = fft(precision)
    return x

def wienerop(N):
    x = laplaceop(N)
    x[abs(x)<1e-5]=1
    sqrtcov = 1/sqrt(x)
    return sqrtcov

def diffuseop(N,sigma):
    kernel = exp(-.5/(sigma**2)*(np.arange(N)-N//2)**2)
    kernel /= sum(kernel)
    kernel = np.roll(kernel,N//2)
    return fft(kernel).real

def flatcov(covariance):
    # Assuming time-independence, get covariance in time
    N = covariance.shape[0]
    sums = zeros(N)
    for i in range(N):
        sums += np.roll(covariance[i],-i)
    return sums / N

def delta(N):
    # Get discrete delta but make it even
    delta = zeros((N,))
    if (N%2==1):
        delta[N//2]=1
    else:
        delta[N//2+1]=delta[N//2]=0.5
    x = fft(delta)
    return x

def differentiator(N):
    # Fourier space discrete differentiaion
    delta = zeros((N,))
    delta[0]=-1
    delta[-1]=1
    x = fft(delta)
    return x

def integrator(N):
    # Fourier space discrete differentiaion
    delta = differentiator(N)
    delta[abs(delta)<1e-10]=1
    return 1./delta

def covfrom(covariance):
    L = len(covariance)
    covmat = zeros((L,L))
    for i in range(L):
        covmat[i,:] = np.roll(covariance,i)
    return covmat

def oucov(ssvar,tau,L)
    # Get covariance structure of process
    covariance = ssvar*exp(-abs(arange(L)-L//2)/tau)
    covariance = np.roll(covariance,L//2+1)
    return covariance
