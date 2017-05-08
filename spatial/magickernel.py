#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
Utilities related to the image interpolation kernel described here
http://johncostella.webs.com/magic/
'''

import numpy as np
from numpy import *

def continuum_kernel(x):
    '''
    limit of continuum magic kernel
    as a piecewise function
    '''
    x = float64(abs(x))
    '''
    if x>1.5: return 0
    if x>0.5: return 0.5*(x-1.5)**2
    return 0.75-x**2
    '''
    return piecewise(x,[x>=1.5,(x>=0.5)&(x<1.5),(x>=0.0)&(x<0.5)],[lambda x:0, lambda x:0.5*(x-1.5)**2, lambda x:0.75-x**2])

def log_spline_basis(N=range(1,6),t=np.arange(100),base=2,offset=1):
    s = log(t+offset)/log(base)
    kernels = array([continuum_kernel(s-k) for k in N]) # evenly spaced in log-time
    kernels = kernels/log(base)/(offset+t) # correction for change of variables, kernals integrate to 1 now
    return kernels

def cosine_kernel(x):
    '''
    raised cosine basis kernel, normalized such that it integrates to 1
    centered at zero. Time is rescaled so that the kernel spans from
    -2 to 2
    '''
    x = float64(abs(x))/2.0*pi
    return piecewise(x,[x<=pi],[lambda x:(cos(x)+1)/4.0])

def log_cosine_basis(N=range(1,6),t=np.arange(100),base=2,offset=1):
    '''
    Generate overlapping log-cosine basis elements
    
    Parameters
    ----------
    N : array of wave quarter-phases
    t : time base
    base : exponent base
    offset : leave this set to 1 (default)
    
    Returns
    -------
    B : array, Basis with n_elements x n_times shape
    '''
    s = log(t+offset)/log(base)
    kernels = array([cosine_kernel(s-k) for k in N]) # evenly spaced in log-time
    kernels = kernels/log(base)/(offset+t) # correction for change of variables, kernels integrate to 1 now
    return kernels

def derive_log_cosine_basis(N,L,min_interval):
    '''
    Build N logarightmically spaced cosine basis functions
    spanning L samples, with a peak resolution of min_interval
    
    # Solve for a time basis with these constraints
    # t[0] = 0
    # t[min_interval] = 1
    # log(L)/log(b) = n_basis+1
    # log(b) = log(L)/(n_basis+1)
    # b = exp(log(L)/(n_basis+1))
    
    Returns
    -------
    B : array, Basis with n_elements x n_times shape
    '''
    t = arange(L)/min_interval+1
    b = exp(log(t[-1])/(N+1))
    B = log_cosine_basis(arange(N),t,base=b,offset=0)
    return B

def exponential_basis(N=range(1,6),t=np.arange(100),base=2,offset=1):
    means = base**array(N)
    t = float64(t)
    kernels = array([exp(-t/m) for m in means])
    kernels = kernels.T / sum(kernels,1)
    return kernels.T

def diffusion_basis(N=range(1,6),t=np.arange(100)):
    '''
    Note: conceptually similar to other basis functions in this file
    with base=2 and offset=1
    repeatly convolves exponential with itself to generate basis
    '''
    print('THIS IS BAD')
    assert 0
    normalize = lambda x:x/sum(x)
    first = fft(exp(-t))
    kernels = [normalize(real(ifft(first**(2**(1+(n-1)*0.5))))) for n in N]
    return array(kernels)

def iterative_orthogonalize_basis(B):
    '''
    iterated orthogonalization to try to help maintain locality?
    as opposed to multiplying by inverse square root B B'
    '''
    B = array(B) # acting in place so make a copy
    for i in range(1,shape(B)[0]):
        a = B[i-1]
        b = B[i]
        overlap = dot(a,b)
        B[i] -= a*overlap
    return B
