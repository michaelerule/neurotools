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
Utilities related to spatial kernels
'''

import numpy as np
from scipy.signal import convolve2d

def laplace_kernel():
    '''
    Returns a 3x3 laplacian kernel that is as radially 
    symmetric as possible.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    return np.array([[  0.5,   2. ,   0.5],
       [  2. , -10. ,   2. ],
       [  0.5,   2. ,   0.5]])/3.

def laplacian(x):
    '''
    Graph laplacian of a 2D mesh with absorbing boundary
        
    In the middle
        0  1  0
        1 -4  1
        0  1  0
    
    At edges
         1  0
        -3  1
         1  0
    
    At corners
        -2  1
         1  0

    Example
    -------
    >>> test = np.zeros((5,11),'float32')
    >>> test[2,5] = 1
    >>> showim(test)
    >>> showim(laplacian(test))
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    n,m = x.shape
    # Middle cases
    result = np.copy(x)*-4
    # Edge cases
    result[ 0, :] = x[0,:]*-3
    result[ :, 0] = x[:,0]*-3
    result[-1, :] = x[-1,:]*-3
    result[ :,-1] = x[:,-1]*-3
    # Corner cases
    result[ 0, 0] = x[ 0, 0]*-2
    result[ 0,-1] = x[ 0,-1]*-2
    result[-1, 0] = x[-1, 0]*-2
    result[-1,-1] = x[-1,-1]*-2
    # Add neighbors
    result[1: , :]   += x[ :-1,:]
    result[:  ,1:]   += x[ :,  :-1]
    result[:-1, :]   += x[1:,  :]
    result[:  , :-1] += x[ :, 1:]
    return result

def gaussian_2D_kernel(sigma):
    '''
    Generate 2D Gaussian kernel as product of 2 1D kernels
    >>> showim(gaussian_2D_kernel(1))
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    radius  = int(np.ceil(sigma*3))
    support = 1+2*radius
    kern_1D = np.exp(-np.arange(-radius,radius+1)**2/(2*sigma**2))
    kernel = kern_1D[:,None]*kern_1D[None,:]
    kernel /= np.sum(kernel)
    return kernel

def absorbing_gaussian(x,sigma):
    '''
    Applies a gaussian convolution to 2d array `x` with absorbing
    boundary conditions.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    support = 1+sigma*6
    normalization = np.zeros(x.shape,'double')
    result = np.zeros(x.shape,'double')
    kernel = gaussian_2D_kernel(sigma)
    return convolve2d(x, kernel, mode='same', boundary='symm')

def laplace_kernel():
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    return np.array([[  0.5,   2. ,   0.5],
       [  2. , -10. ,   2. ],
       [  0.5,   2. ,   0.5]])/3.

def absorbing_laplacian(x):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    kernel = laplace_kernel()
    return np.convolve2d(x, kernel, mode='same', boundary='symm')
    
def continuum_kernel(x):
    '''
    limit of continuum magic kernel as a piecewise function.
    See http://johncostella.webs.com/magic/
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    x = np.float64(abs(x))
    '''
    if x>1.5: return 0
    if x>0.5: return 0.5*(x-1.5)**2
    return 0.75-x**2
    '''
    return np.piecewise(x,[x>=1.5,(x>=0.5)&(x<1.5),(x>=0.0)&(x<0.5)],\
        [lambda x:0, lambda x:0.5*(x-1.5)**2, lambda x:0.75-x**2])

def log_spline_basis(N=range(1,6),t=np.arange(100),base=2,offset=1):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    s = np.log(t+offset)/np.log(base)
    kernels = np.array([continuum_kernel(s-k) for k in N]) # evenly spaced in log-time
    kernels = kernels/np.log(base)/(offset+t) # correction for change of variables, kernals integrate to 1 now
    return kernels

def cosine_kernel(x):
    '''
    raised cosine basis kernel, normalized such that it integrates to 1
    centered at zero. Time is rescaled so that the kernel spans from
    -2 to 2
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    x = np.float64(np.abs(x))/2.0*pi
    return np.piecewise(x,[x<=pi],[lambda x:(np.cos(x)+1)/4.0])

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
    s = np.log(t+offset)/np.log(base)
    # evenly spaced in log-time
    kernels = np.array([cosine_kernel(s-k) for k in N]) 
    # correction for change of variables, kernels integrate to 1 now
    kernels = kernels/np.log(base)/(offset+t) 
    return kernels

def make_cosine_basis(N,L,min_interval):
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
    t = np.arange(L)/min_interval+1
    b = np.exp(np.log(t[-1])/(N+1))
    B = log_cosine_basis(np.arange(N),t,base=b,offset=0)
    return B

def exponential_basis(N=range(1,6),t=np.arange(100),base=2,offset=1):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    means = base**np.array(N)
    t = np.float64(t)
    kernels = np.array([np.exp(-t/m) for m in means])
    kernels = kernels.T / np.sum(kernels,1)
    return kernels.T

def diffusion_basis(N=range(1,6),t=np.arange(100)):
    '''
    Note: conceptually similar to other basis functions in this file
    with base=2 and offset=1
    repeatly convolves exponential with itself to generate basis
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    print('THIS IS BAD')
    assert 0
    normalize = lambda x:x/np.sum(x)
    first = np.fft(np.exp(-t))
    kernels = [normalize(np.real(np.ifft(first**(2**(1+(n-1)*0.5))))) for n in N]
    return np.array(kernels)

def iterative_orthogonalize_basis(B):
    '''
    iterated orthogonalization to try to help maintain locality?
    as opposed to multiplying by inverse square root B B'
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    B = np.array(B) # acting in place so make a copy
    for i in range(1,B.shape[0]):
        a = B[i-1]
        b = B[i]
        overlap = np.dot(a,b)
        B[i] -= a*overlap
    return B

