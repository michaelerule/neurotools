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
import sys
# more py2/3 compat
from neurotools.system import *
if sys.version_info<(3,):
    from itertools import imap as map
# END PYTHON 2/3 COMPATIBILITY BOILERPLATE

'''
Collecting matrix-related subroutines
'''

import numpy as np

def triu_elements(M):
    '''
    Somewhat like matlab's "diag" function, but for upper triangular matrices
    
    triu_elements(randn(D*(D+1)//2))
    '''
    if len(M.shape)==2:
        # M is a matrix
        if not M.shape[0]==M.shape[1]:
            raise ValueError("Extracting upper triangle elements supported only on square arrays")
        # Extract upper trianglular elements
        i = np.triu_indices(M.shape[0])
        return M[i]
    if len(M.shape)==1:
        # M is a vector
        # N(N+1)/2 = K
        # N(N+1) = 2K
        # NN+N = 2K
        # NN+N-2K=0
        # A x^2 + Bx + C
        # -1 +- sqrt(1-4*1*(-2K))
        # -----------------------
        #           2
        # 
        # (sqrt(1+8*K)-1)/2
        K = M.shape[0]
        N = (np.sqrt(1+8*K)-1)/2
        if N!=round(N):
            raise ValueError('Cannot pack %d elements into a square triangular matrix'%K)
        N = int(N)
        result = np.zeros((N,N))
        result[np.triu_indices(N)] = M
        return result
    raise ValueError("Must be 2D matrix or 1D vector")

def tril_elements(M):
    '''
    Somewhat like matlab's "diag" function, but for lower triangular matrices
    
    tril_elements(randn(D*(D+1)//2))
    '''
    if len(M.shape)==2:
        # M is a matrix
        if not M.shape[0]==M.shape[1]:
            raise ValueError("Extracting upper triangle elements supported only on square arrays")
        # Extract upper trianglular elements
        i = np.tril_indices(M.shape[0])
        return M[i]
    if len(M.shape)==1:
        # M is a vector
        # N(N+1)/2 = K
        # N(N+1) = 2K
        # NN+N = 2K
        # NN+N-2K=0
        # A x^2 + Bx + C
        # -1 +- sqrt(1-4*1*(-2K))
        # -----------------------
        #           2
        # 
        # (sqrt(1+8*K)-1)/2
        K = M.shape[0]
        N = (np.sqrt(1+8*K)-1)/2
        if N!=round(N):
            raise ValueError('Cannot pack %d elements into a square triangular matrix'%K)
        N = int(N)
        result = np.zeros((N,N))
        result[np.tril_indices(N)] = M
        return result
    raise ValueError("Must be 2D matrix or 1D vector")

def column(x):
    '''
    Ensure that x is a column vector
    if x is multidimensional, x.ravel() will be calle first
    '''
    x = x.ravel()
    return x.reshape((x.shape[0],1))
    
def row(x):
    '''
    Ensure that x is a row vector
    '''
    x = x.ravel()
    return x.reshape((1,x.shape[0]))

def rcond(x):
    return 1./np.linalg.cond(x)

def check_finite_real(M):
    if np.any(~np.isreal(M)):
        raise ValueError("Complex value encountered for real vector")
    if np.any(~np.isfinite(M)):
        raise ValueError("Non-finite number encountered")

def check_covmat(C,N=None,eps=1e-6):
    '''
    Verify that matrix M is a size NxN precision or covariance matirx
    '''
    if not type(C)==np.ndarray:
        raise ValueError("Covariance matrix should be a 2D numpy array")
    if not len(C.shape)==2:
        raise ValueError("Covariance matrix should be a 2D numpy array")
    if N is None: 
        N = C.shape[0]
    if not C.shape==(N,N):
        raise ValueError("Expected size %d x %d matrix"%(N,N))
    if np.any(~np.isreal(C)):
        raise ValueError("Covariance matrices should not contain complex numbers")
    if np.any(~np.isfinite(C)):
        raise ValueError("Covariance matrix contains NaN or ±inf!")
    if not np.all(np.abs(C-C.T)<eps):
        raise ValueError("Covariance matrix is not symmetric up to precision %0.1e"%eps)
    w,v = np.linalg.eig(C)
    if np.any(np.abs(np.imag(w))>eps):
        raise ValueError('Covariance should not have complex eigenvalues')
    # Allow small imaginary parts due to numeric error, but remove them
    w = np.real(w)
    if np.any(w<-eps):
        raise ValueError('Covariance matrix contains eigenvalue %0.3e<%0.3e'%(np.min(w),-eps)) 
    # trucate spectrum at some small value
    w[w<eps]=eps
    # Very large eigenvalues can also cause numeric problems
    w[w>1./eps]=1./eps;
    #maxe = np.max(np.abs(w))
    #if maxe>10./eps:
    #    raise ValueError('Covariance matrix eigenvalues %0.2e is larger than %0.2e'%(maxe,10./eps))
    # Rebuild matrix
    C = v.dot(diag(w)).dot(v.T)
    # Ensure symmetry (only occurs as a numerical error for very large matrices?)
    C = 0.5*(C+C.T)
    return C

def real_eig(M,eps=1e-9):
    if not (type(M)==np.ndarray):
        raise ValueError("Expected array; type is %s"%type(M))
    if np.any(np.abs(np.imag(M))>eps):
        raise ValueError("Matrix has imaginary values >%0.2e; will not extract real eigenvalues"%eps)
    M = np.real(M)
    w,v = np.linalg.eig(M)
    if np.any(abs(np.imag(w))>eps):
        raise ValueError('Eigenvalues with imaginary part >%0.2e; matrix has complex eigenvalues'%eps)
    w = np.real(w)
    order = np.argsort(w)
    w = w[order]
    v = v[:,order]
    return w,v

def logdet(C,eps=1e-6,safe=0):
    if safe: C = check_covmat(C,eps=eps)
    w = np.linalg.eig(C)[0]
    w = np.real(w)
    w[w<eps]=eps
    det = np.sum(np.log(w))
    return det