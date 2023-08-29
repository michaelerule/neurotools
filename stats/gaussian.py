#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Routines for working with scalar and multivariate Gaussian
distributions in filtering applicationss
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import scipy
import numpy as np
from numpy import pi
import scipy.linalg
import numpy.random
import random
from neurotools.linalg.matrix import check_covmat, check_covmat_fast, check_finite_real, logdet
from neurotools.linalg.matrix import real_eig
from scipy.optimize import minimize
from neurotools.util.functions import slog,sexp
from neurotools.stats.distributions import poisson_logpdf

def gaussian_quadrature(p,domain,eps=1e-12):
    '''
    Numeric integration of probability density `p` over 
    `domain`; Values in `p` smaller than `eps` (defaults to
    1e-12) are set to `eps`.
    
    Treat f as a density and estimate it's mean and 
    precision over the domain.
    
    Parameters
    ----------
    p:
        1D iterable of probabilities
    domain:
        1D iterable of domain points corresponding to `p`
    eps: positive float; default 1e-12
        Minimum probability value permitted
    
    Returns
    -------
    Gaussian
        Gaussian object with mean and precision matching 
        estimated mean and precision of distribution 
        specified by (`p`,`domain`).
    '''
    assert np.all(np.isfinite(p))
    p = np.array(p)
    p[p<eps] = eps   
    p/= np.sum(p)
    m = np.sum(domain*p)
    assert np.isfinite(m)
    v = np.sum((domain-m)**2*p)
    assert np.isfinite(v)
    t = 1./(v+1e-10)
    assert np.isfinite(t)
    assert t>=0
    return Gaussian(m, t)

def gaussian_quadrature_logarithmic(logp,domain):
    '''
    Numeric integration of log-probability. Not yet implemented.
    
    Parameters
    ----------
    p
        1D iterable of probabilities
    domain
        1D iterable of domain points corresponding to `p`
    
    Returns
    -------
    Gaussian
    '''
    raise NotImplementedError('This function is not yet implemented')
    normalization = sum(p)
    m = np.sum(domain*p)/normalization
    assert np.isfinite(m)
    v = np.sum((domain-m)**2*p/normalization)
    assert np.isfinite(v)
    t = 1./(v+1e-10)
    assert np.isfinite(t)
    assert t>=0
    return Gaussian(m, t)

class Gaussian:
    '''
    Scalar Gaussian model to use in abstracted forward-backward
    Supports multiplication of Gaussians

    Attributes
    ----------
    m: float
        mean 
    t: float
        precision (reciprocal of variance)
    '''
    def __init__(s,m,t):
        '''
    
        Parameters
        ----------
        m : float
            Mean of distribution
        t : float
            Precision (1/variance) of distribution
        
        Returns
        -------
        Gaussian
            Gaussian density object
        '''
        s.m,s.t = m,t
        s.lognorm = 0
    def __mul__(s,o):
        '''
        Multiply two Gaussians together in terms of 
        probability, for example to apply Bayes rule. 
    
        Parameters
        ----------
        o : Gaussian
            Another Gaussian. 
            Can also be 1 to indicate the identity
        
        Returns
        -------
        Gaussian 
            New gaussian distribution that is the product of
            this distribution with the distribution `o`
        '''
        if o==1: return s
        assert isinstance(o,Gaussian)
        t = s.t+o.t
        m = (s.m*s.t + o.m*o.t)/(t) if abs(t)>1e-16 else s.m+o.m        
        assert np.isfinite(t)
        assert np.isfinite(m)
        result = Gaussian(m,t)
        # propagating normalization factors for forward-backward
        if hasattr(o,'lognorm'):
            result.lognorm = s.lognorm+o.lognorm
        else:
            result.lognorm = s.lognorm
        return result
    def __truediv__(s,o):
        '''
        Divide this Gaussian distribtion by another 
        Gaussian distribution `o`.
    
        Parameters
        ----------
        o : Gaussian
            Another Gaussian. Can also be 1 to indicate the identity
        
        Returns
        -------
        Gaussian 
            New gaussian distribution reflecting division of this Gaussian
            distribution by the distribution `o`
        '''
        if o==1: return s
        assert isinstance(o,Gaussian)
        t = s.t-o.t
        m = (s.m*s.t - o.m*o.t)/(t) if abs(t)>1e-16 else s.m-o.m
        assert np.isfinite(t)
        assert np.isfinite(m)
        result = Gaussian(m,t)
        # propagating normalization factors for forward-backward
        if hasattr(o,'lognorm'):
            result.lognorm = s.lognorm-o.lognorm
        else:
            result.lognorm = s.lognorm
        return result
    __div__  = __truediv__
    def __call__(s,x): 
        '''
        Evaluate this Gaussian PDF at location `x`
    
        Parameters
        ----------
        x : float or np.array
            Point or array of points at which to evaluate the PDF
        
        Returns
        -------
        np.array 
            PDF value at locations specified by `x`
        '''
        return sexp(-0.5*s.t*(x-s.m)**2)*np.sqrt(s.t/(2*np.pi))
    def __str__(s):
        return 'm=%0.4f, t=%0.4f'%(s.m,s.t)
    def logpdf(s,x):
        '''
        The log-pdf of a univariate Gaussian
    
        Parameters
        ----------
        x : float or np.array
            Point or array of points at which to evaluate the log-PDF
        
        Returns
        -------
        np.array 
            log-PDF value at locations specified by `x`
        '''
        return -0.5*s.t*(x-s.m)**2 + 0.5*slog(s.t)-0.91893853320467267#-0.5*log(2*np.pi)
        
        
        
        
        
def MVG_check(M,C,eps=1e-6):
    '''
    Checks that a mean and covariance (or precision) represented a
    valid multivariate Gaussian distribution. The mean must be finite
    and real-valued. The covariance (or precision) matrix must be
    symmetric positive definite.
    
    Parameters
    ----------
    M: 1D np.array
        Mean vector
    C: 2D np.array
        Covariance matrix

    
    Other Parameters
    ----------------
    eps: positive float; default 1e-6
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
    P = np.exp(logP)
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
        return M[:,None]+Q.dot(np.random.randn(len(M),N))
    if C is None:
        # Use precision
        if safe:
            MVG_check(M,P)
        w,v = real_eig(P)
        Q   = v.dot(np.diag((w*(w>0))**-0.5)).dot(v.T)
        return M[:,None]+Q.dot(np.random.randn(len(M),N))

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
    
