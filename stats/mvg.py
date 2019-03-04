#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

'''
Routines for manipulating multivariate Gaussians.
'''

import numpy as np
import scipy
import scipy.linalg
import numpy.random
from   numpy.random import randn
from   numpy import pi

from neurotools.linalg.matrix import check_covmat, check_covmat_fast, check_finite_real, logdet
from neurotools.stats.Gaussian import *
# TODO fix imports
#from neurotools.matrix import *
from neurotools.linalg.matrix import real_eig

def MVG_check(M,C,eps=1e-6):
    '''
    Checks that a mean and covariance (or precision) represented a
    valid multivariate Gaussian distribution. The mean must be finite
    and real-valued. The covariance (or precision) matrix must be
    symmetric positive definite.
    '''
    check_finite_real(M)
    check_covmat(C,len(M),eps=eps)

def MVG_logPDF(X,M,P=None,C=None):
    '''
    N: dimension of distribution
    K: number of samples

    Args:
        X : KxN vector of samples for which to compute the PDF
        M : N mean vector
        P : NxN precision matrix OR
        C : NxN covariance matirx
    
    Example:
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
        xMP  = np.linalg.lstsq(C,xM.T,rcond=None)[0].T
    if C is None:
        # Use precision
        MVG_check(M,P)
        norm = normd+0.5*logdet(P)
        xMP  = xM.dot(P)
    logpr = -0.5*np.sum(xMP*xM,axis=1)
    return norm+logpr

def MVG_PDF(X,M,P=None,C=None):
    '''
    Args:
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

def MVG_sample(M,P=None,C=None,N=1,safe=1):
    '''
    Sample from multivariate Gaussian
    
    Args:
        M : vector mean
        C : covariance matrix
    '''
    if (P is None and C is None):
        raise ValueError("Either a Covariance or Precision matrix is needed")
    if P is None:
        # Use covariance
        if safe:
            MVG_check(M,C)
        w,v = real_eig(C)
        Q   = v.dot(np.diag(np.sqrt(w*(w>0)))).dot(v.T)
        return M[:,None]+Q.dot(randn(len(M),N))
    if C is None:
        # Use precision
        if safe:
            MVG_check(M,P)
        w,v = real_eig(P)
        Q   = v.dot(np.diag((w*(w>0))**-0.5)).dot(v.T)
        return M[:,None]+Q.dot(randn(len(M),N))

def MVG_multiply(M1,P1,M2,P2,safe=1):
    '''
    Multiply two multivariate Gaussians based on precision
    '''
    if safe:
        MVG_check(M1,P1)
        MVG_check(M2,P2)
    P = P1 + P2
    M = scipy.linalg.lstsq(P,np.squeeze(P1.dot(M1)+P2.dot(M2)))[0]
    if safe:
        MVG_check(M,P)
    return M,P


def MVG_multiply_C(M1,C1,M2,C2,safe=1):
    '''
    Multiply two multivariate Gaussians based on covariance
    not implemented
    '''
    if safe:
        MVG_check(M1,C1)
        MVG_check(M2,C2)
    assert 0 
    

def MVG_divide(M1,P1,M2,P2,eps=1e-6,handle_negative='repair',verbose=0):
    '''
    Divide two multivariate Gaussians based on precision
    
    Parameters
    ----------
    handle_negative : 
        'repair' (default): returns a nearby distribution with positive variance
        'ignore': can return a distribution with negative variance
        'error': throws a ValueError if negative variances are producted
    '''
    MVG_check(M1,P1,eps=eps)
    MVG_check(M2,P2,eps=eps)
    P = P1 - P2
    w,v = real_eig(P)
    if any(w<eps):
        if handle_negative=='repair':
            w[w<eps]=eps
            P = v.dot(np.diag(w)).dot(v.T)
            if verbose:
                print('Warning: non-positive precision in Gaussian division')
        elif handle_negative=='ignore':
            pass
        elif handle_negative=='error':
            raise ValueError('Distribution resulting from division has non-positive precision!')
        else:
            raise ValueError('Argument handle_negative must be "repair", "ignore", or "error"')
    M = scipy.linalg.lstsq(P,P1.dot(M1) - P2.dot(M2))[0]
    MVG_check(M,P,eps=eps)
    return M,P

def MVG_projection(M,C,A):
    '''
    Compute a new multi-variate gaussian reflecting distribution of a projection A
    
    Args:
        M : length N vector of the mean
        C : NxN covariance matrix
        A : KxN projection of the vector space (should be unitary?)
    '''
    MVG_check(M,C)
    M = A.dot(M)
    C = scipy.linalg.lstsq(A.T,C.dot(A.T))[0].T
    MVG_check(M,C)
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
    
def MVG_DKL(M0,P0,M1,P1):
    '''
    KL divergence between two Gaussians
    
    Example
    -------
        M = randn(10)
        Q = randn(N,N)
        P = Q.dot(Q.T)
        MGV_DKL(M,P,M,P)
    '''
    MVG_check(M0,P0)
    MVG_check(M1,P1)
    N = len(M0)
    M1M0 = M1-M0
    return 0.5*(np.sum(P1*pinv(P0))+logdet(P0)-logdet(P1)-N+M1M0.T.dot(P1).dot(M1M0))
    #return 0.5*(np.sum(np.diag(P1.dot(np.linalg.pinv(P0))))+logdet(P0)-logdet(P1)-N+M1M0.T.dot(P1).dot(M1M0))
    
def MVG_DKL_CP(M0,C0,M1,P1):
    '''
    KL divergence between two Gaussians
    First one specified using covariance
    Second one using precision
    
    Example
    -------
        M = randn(10)
        Q = randn(N,N)
        P = Q.dot(Q.T)
        MGV_DKL(M,P,M,P)
    '''
    MVG_check(M0,C0)
    MVG_check(M1,P1)
    N = len(M0)
    M1M0 = M1-M0
    return 0.5*(sum(P1*C0)-logdet(C0)-logdet(P1) - N + sum(M1M0.T.dot(P1)*M1M0,axis=0))

def MVG_conditional(M0,P0,M1,P1):
    '''
    If M0,P0 is a multivariate Gaussian
    and M1,P1 is a conditional multivariate Gaussian
    This function returns the joint density
    
    NOT IMPLEMENTED
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
    MVG_check(M,C)
    check_covmat(Q)
    M = A.dot(M)
    C = A.dot(C).dot(A.T) + Q
    MVG_check(M,C)
    return M,C
    
def MVG_kalman_P_inverseA(M,P,A,invA,Q):
    '''
    Performs a Kalman update with linear transform A and covariance Q
    Returns the posterior mean and covariance
    
    This one accepts and returns precision
    This one needs the inverse of the forward state transition matrix
    
    Example
    -------
        C2 = ACA'+Q
        P2 = inv[A inv(P) A' + Q]
        P2 = inv[ A (inv(P) + inv(A) Q inv(A') ) A' ]
        P2 = inv[ A inv(P) (1 + P inv(A) Q inv(A') ) A' ]
        P2 = inv(A') inv(1 + P inv(A) Q inv(A')) P inv(A)
    '''
    MVG_check(M,P)
    check_covmat(Q)
    M = A.dot(M)
    F = eye(len(M)) + P.dot(invA).dot(Q).dot(invA.T)
    G = scipy.linalg.lstsq(F,P)[0]
    P = invA.T.dot(G).dot(invA)
    # re-symmetrizing of covariance matrix seems necessary
    P = 0.5*(P+P.T)
    MVG_check(M,P)
    return M,P

def MVG_kalman_joint(M,C,A,Q,safe=0):
    '''
    Performs a Kalman update with linear transform A and covariance Q
    Keeps track of the joint distribution between the prior and posterior
    '''
    if safe: 
        MVG_check(M,C)
        check_covmat(Q)
    M1 = A.dot(M)
    AC = A.dot(C)
    C1 = AC.dot(A.T) + Q
    if safe: 
        MVG_check(M1,C1)
    M2 = np.concatenate([M,M1])
    C2 = np.array(np.bmat([[C,AC.T],[AC,C1]]))
    if safe: 
        MVG_check(M2,C2)
    return M2,C2

def MVG_kalman_joint_P(M,P,A,Q=None,W=None,safe=0):
    '''
    Performs a Kalman update with linear transform A and covariance Q
    Keeps track of the joint distribution between the prior and posterior
    Accepts and returns precision matrix
    Q must be invertable
    '''
    if safe: 
        check_covmat(P)
    if Q is None and W is None:
        raise ValueError("Please provide either noise covariance Q or its inverse W as arguments")
    if safe: 
        MVG_check(M,P)
    if W is None:
        # Invert noise matrix
        if safe: 
            check_covmat(Q)
        W = pinv(Q)
    # (else use provided inverse, but check it)
    if safe: 
        W = check_covmat(W)
    M2  = np.concatenate([M,A.dot(M)])
    nWA = -W.dot(A)
    P2 = np.array(np.bmat([
        [P-nWA.T.dot(A),nWA.T],
        [nWA           , W]]))
    if safe: 
        MVG_check(M2,P2)
    return M2,P2

