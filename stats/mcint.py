#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Routines for monte-carlo integration
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import sys

def monte_carlo_expectation(f,maxiter=int(1e6),converge=1e-2,verbose=False):
    '''
    x = monte_carlo_expectation(f,maxiter,converge)
    
    Evaluate expectation of f using Monte-Carlo integration.
    For simplicit (for now), this casts the return value of f() to a float64 array
    Ensure the return value is compatible with this datatype.
    This uses the standard error of the mean to check for convergence. 
    It converges slowly at 1/sqrt(n)

    Example:
    
    .. code-block:: python
    
        def f():
            x = randn(2)+array([9,-9])
            return x
        Ex = monte_carlo_moments(f,verbose=1,maxiter=100000,converge=1e-2)
        print('Ex:\\n',Ex)
    
    Parameters
    ----------
    f: 
        function that returns array_like.
    maxiter: 
        maximum number of samples to draw
    converge: 
        maximum absolute error tolerated
    
    Returns
    -------
    number or array-like:
        Estimate of the mean of f
    '''
    # perform argument validation
    maxiter=int(maxiter)
    assert(maxiter>0)
    assert(converge>0)
    # perform checks for numerical accuracy
    dtype     = np.float64
    eps       = np.sqrt(np.finfo(dtype).eps)
    maxsample = 1./eps
    if (maxiter>maxsample):
        print('Warning: maximum iterations cannot be computed with acceptable precision')
        assert 0
    # draw the first sample
    nsamp = 1.0
    sample   = dtype(f()).ravel()
    moment_1 = sample
    variance = np.mean(sample**2)
    #moment_2 = np.outer(sample,sample)
    if verbose:
        sys.stdout.write('\n\r')
        sys.stdout.flush()
    # draw samples until we have converged
    for i in range(maxiter):
        sample   = dtype(f()).ravel()
        moment_1 = moment_1*(nsamp/(nsamp+1.0))+(1.0/(nsamp+1.0))*sample
        variance = variance*(nsamp/(nsamp+1.0))+(1.0/(nsamp+1.0))*np.mean(sample**2)
        stderror = np.sqrt((variance-np.mean(moment_1**2))/nsamp)
        assert np.isfinite(stderror)
        if (stderror<=converge): break
        nsamp += 1
        if verbose and (nsamp%100==0):
            sys.stdout.write(('\r\b'*40)+'Sample %d, error %0.2f'%(nsamp,stderror))
            sys.stdout.flush()
    if verbose: 
        sys.stdout.write('\n')
        sys.stdout.flush()
    return moment_1


def monte_carlo_moments(f,maxiter=int(1e6),converge=1e-2,verbose=False):
    ''' 
    x = monte_carlo_expectation(f,maxiter,converge)
    
    Evaluate expectation of f using Monte-Carlo integration.
    For simplicit (for now), this casts the return value of f() to a float64 array
    Ensure the return value is compatible with this datatype.
    This uses the standard error of the mean to check for convergence. 
    It converges very slowly (1/sqrt(n)), so don't ask for too much precision.

    Example:
    
    .. code-block:: python
    
        def f():
            x = randn(2)+array([9,-9])
            return x
        Ex,Exx = monte_carlo_moments(f,verbose=1,maxiter=100000,converge=1e-2)
        print('Ex:\\n',Ex)
        print('Exx:\\n',Exx)
    
    Parameters
    ----------
    f : 
        function that returns array_like.
    maxiter : 
        maximum number of samples to draw
    converge : 
        maximum absolute error tolerated
    
    Returns
    -------
    Estimate of the mean and second moment of f
    '''
    # perform argument validation
    maxiter=int(maxiter)
    assert(maxiter>0)
    assert(converge>0)
    # perform checks for numerical accuracy
    dtype     = np.float64
    eps       = np.sqrt(np.finfo(dtype).eps)
    maxsample = 1./eps
    if (maxiter>maxsample):
        print('Warning: maximum iterations cannot be computed with acceptable precision')
        assert 0
    # draw the first sample
    nsamp = 1.0
    sample   = dtype(f()).ravel()
    moment_1 = sample
    moment_2 = np.outer(sample,sample)
    if verbose:
        sys.stdout.write('\n\r')
        sys.stdout.flush()
    # draw samples until we have converged
    for i in range(maxiter):
        sample   = dtype(f()).ravel()
        moment_1 = moment_1*(nsamp/(nsamp+1.0))+(1.0/(nsamp+1.0))*sample
        moment_2 = moment_2*(nsamp/(nsamp+1.0))+(1.0/(nsamp+1.0))*np.outer(sample,sample)
        
        # check error for convergence
        covariance          = moment_2 - np.outer(moment_1,moment_1)
        standard_error      = covariance / nsamp
        mean_standard_error = np.sqrt(np.trace(standard_error)/sample.shape[0])
        assert np.isfinite(mean_standard_error)
        if (mean_standard_error<=converge):
            break
        nsamp += 1
        if verbose and (nsamp%100==0):
            sys.stdout.write(('\r\b'*40)+'Sample %d, error %0.2f'%(nsamp,mean_standard_error))
            sys.stdout.flush()
    if verbose: 
        sys.stdout.write('\n')
        sys.stdout.flush()
    return moment_1,moment_2
