#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import numpy.linalg
import scipy
import scipy.linalg

'''
Functions for generating discrete representations of certain operators. 
Note: This is partially redundant with `neurotools.spatial.kernels`.
'''

def laplaceop(N):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    precision = np.zeros((N,))
    if (N%2==1):
        precision[N//2]=2
        precision[N//2+1]=precision[N//2-1]=-1
    else:
        precision[N//2+1]=precision[N//2]=1
        precision[N//2+2]=precision[N//2-1]=-1
    x = np.fft(precision)
    return x

def wienerop(N):
    '''
    Square-root covariance operator for standard 1D Wiener process
    
    Parameters
    ----------
    N : size of operator
    
    Returns
    -------
    '''
    x = laplaceop(N)
    x[abs(x)<1e-5]=1
    sqrtcov = 1/sqrt(x)
    return sqrtcov

def diffuseop(N,sigma):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    kernel = np.exp(-.5/(sigma**2)*(np.arange(N)-N//2)**2)
    kernel /= np.sum(kernel)
    kernel = np.roll(kernel,N//2)
    return np.fft(kernel).real

def flatcov(covariance):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    # Assuming time-independence, get covariance in time
    N = covariance.shape[0]
    sums = np.zeros(N)
    for i in range(N):
        sums += np.roll(covariance[i],-i)
    return sums / N

def delta(N):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    # Get discrete delta but make it even
    delta = np.zeros((N,))
    if (N%2==1):
        delta[N//2]=1
    else:
        delta[N//2+1]=delta[N//2]=0.5
    x = np.fft(delta)
    return x

def differentiator(N):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    # Fourier space discrete differentiaion
    delta = np.zeros((N,))
    delta[0]=-1
    delta[-1]=1
    x = np.fft(delta)
    return x

def integrator(N):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    # Fourier space discrete differentiaion
    delta = differentiator(N)
    delta[abs(delta)<1e-10]=1
    return 1./delta

def covfrom(covariance):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    L = len(covariance)
    covmat = np.zeros((L,L))
    for i in range(L):
        covmat[i,:] = np.roll(covariance,i)
    return covmat

def oucov(ssvar,tau,L):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    # Get covariance structure of process
    covariance = ssvar*np.exp(-np.abs(np.arange(L)-L//2)/tau)
    covariance = np.roll(covariance,L//2+1)
    return covariance

def gaussian1DblurOperator(n,sigma):
    '''
    Returns a 1D Gaussan blur operator of size n
    '''
    x   = np.linspace(0,n-1,n); # 1D domain
    tau = 1.0/sigma**2;       # precision
    k   = np.exp(-tau*x**2);    # compute (un-normalized) 1D kernel
    op  = scipy.linalg.special_matrices.toeplitz(k,k);     # convert to an operator from n -> n
    # normalize rows so density is conserved
    op /= np.sum(op)
    # truncate small entries
    big = np.max(op)
    toosmall = 1e-4*big
    op[op<toosmall] = 0
    # (re) normalize rows so density is conserved
    op /= np.sum(op)
    return op

def gaussian2DblurOperator(n,sigma):
    '''
    Returns a 2D Gaussan blur operator for a n x n sized domain
    Constructed as a tensor product of two 1d blurs of size n
    '''
    x   = np.linspace(0,n-1,n) # 1D domain
    tau = 1.0/sigma**2       # precision
    k   = np.exp(-tau*x**2)    # compute (un-normalized) 1D kernel
    tp  = scipy.linalg.special_matrices.toeplitz(k,k)     # convert to an operator from n -> n
    op  = scipy.linalg.special_matrices.kron(tp,tp)       # take the tensor product to get 2D operator
    # normalize rows so density is conserved
    op /= np.sum(op,axis=1)
    # truncate small entries
    big = np.max(op)
    toosmall = 1e-4*big
    op[op<toosmall] = 0
    # (re) normalize rows so density is conserved
    op /= np.sum(op,axis=1)
    return op
    
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
    t = arange(L)/min_interval+1
    b = exp(log(t[-1])/(N+1))
    B = log_cosine_basis(arange(N),t,base=b,offset=0)
    return B
