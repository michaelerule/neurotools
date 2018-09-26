#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
from scipy.signal.signaltools import fftconvolve,hilbert
from scipy.signal import butter, filtfilt, lfilter
from scipy.linalg import lstsq,pinv

def issquare(M):
    return len(M.shape)==2 and M.shape[0]==M.shape[1]

def iscolumn(M):
    return len(M.shape)==2 and M.shape[1]==1

def isrow(M):
    return len(M.shape)==2 and M.shape[0]==1

def asrow(M):
    if isinstance(M,(int,float)):
        M = np.array([[M]])
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    assertfinitereal(M)
    if len(M.shape)==1:
        return M.reshape((1,len(M)))
    if isrow(M): 
        return M
    MT = M.T
    if isrow(MT):
        return MT
    raise ValueError('Cannot cast argument shaped (%s) to row vector'%(M.shape,))

def ascolumn(M):
    if isinstance(M,(int,float)):
        M = np.array([[M]])
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    assertfinitereal(M)
    if len(M.shape)==1:
        return M.reshape((np.size(M),1))
    if iscolumn(M): 
        return M
    MT = M.T
    if iscolumn(MT):
        return MT
    raise ValueError('Cannot cast argument shaped (%s) to column vector'%(M.shape,))

def assquare(M):
    if isinstance(M,(int,float)):
        M = np.array([[M]])
    assertfinitereal(M)
    return assertsquare(M)
    
def assertsquare(M):
    if isinstance(M,(int,float)):
        raise ValueError('Argument should be square array, but is a scalar')
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    assertfinitereal(M)
    if not issquare(M):
        raise ValueError('Argument should be square np.ndarray')
    return M
    
def assertcolumn(M):
    if isinstance(M,(int,float)):
        raise ValueError('Argument should be column vector, but is a scalar')
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    assertfinitereal(M)
    if not iscolumn(M):
        raise ValueError('Argument should be a column vector')
    return M
    
def assertrow(M):
    if isinstance(M,(int,float)):
        raise ValueError('Argument should be row vector, but is a scalar')
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    assertfinitereal(M)
    if not isrow(M):
        raise ValueError('Argument should be a row vector')
    return M

def assertvector(M):
    if isinstance(M,(int,float)):
        raise ValueError('Argument should be a 1-D vector, but is a scalar')
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    assertfinitereal(M)
    if len(M.shape)>1:
        raise ValueError('Argument should be a 1-D vector, but has shape (%s)'%((M.shape),))
    return M

def asvector(M):
    if isinstance(M,(int,float)):
        M = [M]
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    assertfinitereal(M)
    if np.sum(M.shape!=1)>1:
        raise ValueError('More than one dimension longer than 1, cannot cast to 1-D vector')
    M = np.squeeze(M)
    assert(np.size(M) == M.shape[0])
    return M

def scalar(M):
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    if isinstance(M,np.ndarray):
        if not np.size(M)==1:
            raise ValueError('Argument should be a scalar')
        return M.ravel()[0]
    if isinstance(M,(int,float)):
        return M
    try:
        return float(M)
    except:
        pass
    raise ValueError('Argument %s of type %s is not a scalar'%(M,type(M)))
    
def assertfinite(M):
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    if not np.all(np.isfinite(M)):
        raise ValueError('Argument must not contain inf or NaN')
    return M

def assertreal(M):
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    if not np.all(np.isreal(M)):
        raise ValueError('Argument must not contain inf or NaN')
    return M
    
def assertfinitereal(M):
    M = assertfinite(M)
    M = assertreal(M)
    return M
    
def assertshape(M,shape):
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    if not M.shape==shape:
        raise ValueError('Expected shape (%s) but found shape (%s)'%(shape,M.shape))
    return M
    
