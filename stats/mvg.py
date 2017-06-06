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
Routines for manipulating multivariate Gaussians.
'''

import numpy as np
import scipy
import scipy.linalg
import numpy.random
from numpy.random import randn
from numpy import pi

print('init MVG')

def real_eig(M):
    if not np.all(np.isreal(M)):
        raise ValueError("Matrix has complex entries; log-determinate unsupported")
    if not (type(M)==np.ndarray):
        raise ValueError("Expected array; type is %s"%type(M))
    w,v = np.linalg.eig(M)
    if any(np.iscomplex(w)):
        raise ValueError('Complex eigenvalues! Matrix is non-PD or too large for machine precision')
    w = np.real(w)
    order = np.argsort(w)
    w = w[order]
    v = v[:,order]
    return w,v


print('defined real_eig')

def condition(M):
    w,v = real_eig(M)
    mine = np.min(abs(w))
    maxe = np.max(abs(w))
    return mine/maxe

def logdet(M,eps=1e-9):
    w,v = real_eig(M)
    if any(w<eps):
        raise ValueError('Non positive-definite matrix provided, log-determinant is not real-valued') 
    det = np.sum(np.log(w))
    return det

def check_finite_real(M):
    if np.any(~np.isreal(M)):
        raise ValueError("Complex value encountered for real vector")
    if np.any(~np.isfinite(M)):
        raise ValueError("Non-finite number encountered")

def check_covmat(C,N,eps=1e-9):
    '''
    Verify that matrix M is a size NxN precision or covariance matirx
    '''
    if not C.shape==(N,N):
        raise ValueError("Expected size %d x %d matrix"%(N,N))
    w,v = real_eig(C)
    if any(w<eps):
        raise ValueError('Non positive-definite matrix') 

def MVG_check(M,C):
    check_finite_real(M)
    N = len(M)
    check_covmat(C,N)

def MVG_logPDF(X,M,P=None,C=None):
    '''
    X : KxN vector of samples for which to compute the PDF
    M : N mean vector
    P : NxN precision matrix OR
    C : NxN covariance matirx
    
    Test
    ----
    N = 10
    K = 100
    M = randn(10)
    Q = randn(N,N)
    C = Q.dot(Q.T)
    X = randn(K,N)
    MVG_logPDF(X,M,C=C) - MVG_logPDF(X,M,P=np.linalg.pinv(C))
    '''
    N = len(M)
    if (P is None and C is None):
        raise ValueError("Either a Covariance or Precision matrix is needed")
    normd = -0.5*N*np.log(2*pi)
    xM = X-M
    if P is None:
        # Use covariance
        # Compute product with precision (inverse covariance)
        # Using least-squares, rather than inverting the matrix
        MVG_check(M,C)
        norm = normd-0.5*logdet(C)
        xMP  = np.linalg.lstsq(C,xM.T)[0].T
    if C is None:
        # Use precision
        MVG_check(M,P)
        norm = normd+0.5*logdet(P)
        xMP  = xM.dot(P)
    logpr = -0.5*np.sum(xMP*xM,axis=1)
    return norm+logpr

def MVG_PDF(X,M,P=None,C=None):
    '''
    X : NxK vector of samples for which to compute the PDF
    M : N mean vector
    P : NxN precision matrix OR
    C : NxN covariance matirx
    '''
    logP = MVG_logPDF(X,M,P=P,C=C)
    P = exp(logP)
    if np.any(~np.isfinite(P)):
        raise ValueError("Invalid probability; something is wrong?")
    return P

def MVG_sample(M,P=None,C=None,N=1):
    '''
    Sample from multivariate Gaussian
    
    M : vector mean
    C : covariance matrix
    '''
    if (P is None and C is None):
        raise ValueError("Either a Covariance or Precision matrix is needed")
    if P is None:
        # Use covariance
        MVG_check(M,C)
        w,v = real_eig(C)
        Q   = v.dot(diag(sqrt(w*(w>0)))).dot(v.T)
        return M[:,None]+Q.dot(randn(len(M),N))
    if C is None:
        # Use precision
        MVG_check(M,P)
        w,v = real_eig(P)
        Q   = v.dot(diag((w*(w>0))**-0.5)).dot(v.T)
        return M[:,None]+Q.dot(randn(len(M),N))

def MVG_multiply(M1,P1,M2,P2):
    '''
    Multiply two multivariate Gaussians based on precision
    '''
    MVG_check(M1,P1)
    MVG_check(M2,P2)
    P = P1 + P2
    M = scipy.linalg.lstsq(P,np.squeeze(P1.dot(M1)+P2.dot(M2)))[0]
    return M,P


def MVG_multiply_C(M1,C1,M2,C2):
    '''
    Multiply two multivariate Gaussians based on covariance
    not implemented
    '''
    MVG_check(M1,C1)
    MVG_check(M2,C2)
    assert 0 
    

def MVG_divide(M1,P1,M2,P2):
    '''
    Divide two multivariate Gaussians based on precision
    '''
    MVG_check(M1,P1)
    MVG_check(M2,P2)
    P = P1 - P2
    M = scipy.linalg.lstsq(P,P1.dot(M1) - P2.dot(M2))[0]
    return M,P

def MVG_projection(M,C,A):
    '''
    Compute a new multi-variate gaussian reflecting distribution of a projection A
    
    M : length N vector of the mean
    C : NxN covariance matrix
    A : KxN projection of the vector space (should be unitary?)
    '''
    MVG_check(M,C)
    
    M = A.dot(M)
    #C = A.dot(C).dot(A.T)
    #C = A.dot(C).dot(pinv(A))
    C = scipy.linalg.lstsq(A.T,C.dot(A.T))[0].T
    return M,C

def MVG_entropy(M,P=None,C=None):
    '''
    Differential entropy of a multivariate gaussian distribution
    M : N mean vector
    P : NxN precision matrix OR
    C : NxN covariance matirx
    '''
    if (P is None and C is None):
        raise ValueError("Either a Covariance or Precision matrix is needed")
    if P is None:
        # Use covariance
        MVG_check(M,C)
        return 0.5*(k*log(2*pi)+k+logdet(C))
    if C is None:
        # Use precision
        MVG_check(M,P)
        return 0.5*(k*log(2*pi)+k-logdet(P))
    
def MGV_DKL(M0,P0,M1,P1):
    '''
    KL divergence between two Gaussians
    
    Test
    ----
    M = randn(10)
    Q = randn(N,N)
    P = Q.dot(Q.T)
    MGV_DKL(M,P,M,P)
    '''
    MVG_check(M0,P0)
    MVG_check(M1,P1)
    N = len(M0)
    M1M0 = M1-M0
    return 0.5*(np.sum(np.diag(P1.dot(np.linalg.pinv(P0))))+logdet(P0)-logdet(P1)-N+M1M0.T.dot(P1).dot(M1M0))

def MVG_conditional(M0,P0,M1,P1):
    '''
    If M0,P0 is a multivariate Gaussian
    and M1,P1 is a conditional multivariate Gaussian
    This function returns the joint density
    '''
    MVG_check(M0,P0)
    MVG_check(M1,P1)
    N0 = len(M0)
    N1 = len(M1)
    assert 0
    
def MVG_kalman(M,C,A,Q):
    '''
    Performs a Kalman update with linear transform A and covariance Q
    Returns the posterior mean and covariance
    '''
    M = A.dot(M)
    C = A.dot(C).dot(A.T) + Q
    return M,C

def MVG_kalman_joint(M,C,A,Q):
    '''
    Performs a Kalman update with linear transform A and covariance Q
    Keeps track of the joint distribution between the prior and posterior
    '''
    MVG_check(M,C)
    M1 = A.dot(M)
    AC = A.dot(C)
    C1 = AC.dot(A.T) + Q
    MVG_check(M1,C1)
    M2 = np.concatenate([M,M1])
    C2 = np.array(np.bmat([[C,AC],[AC.T,C1]]))
    MVG_check(M2,C2)
    return M2,C2



