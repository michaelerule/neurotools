#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function


'''
Utility routines for exploring Restricted Boltzmann Machines (RBMs)

Under development, intended to be used with internal lab archive format
(not ready for public use)
'''

from collections import defaultdict
import numpy as np

from neurotools.util.functions import slog, sexp, g, f, f1, f2

def bar_f(x):
    '''
    Function mapping activation to 1-probability
    
    Parameters
    ----------
    Returns
    -------
    '''
    return f(-x)

def inv_f(x,eps=1e-12):
    '''
    Inverse of logistic function
    
    Parameters
    ----------
    Returns
    -------
    '''
    x = 1./x-1
    x[x<eps]=eps
    return -slog(x)


a2p = f
a2q = bar_f
p2a = inv_f

def Acond(h,W,Bv):
    '''
    Conditional RBM activations
    
    Parameters
    ----------
    Returns
    -------
    '''
    return (W.dot(h).T + Bv).T

def Pcond(h,W,Bh):
    '''
    Conditional RBM probabilities
    
    Parameters
    ----------
    Returns
    -------
    '''
    return f(Acond(h,W,Bh))

def lnPr(s,p,eps=1e-12,axis=-1):
    '''
    Compute probability of bits s given Bernoulli probabilities p
    Assuming factorized distribution
    \prod p^x (1-p)^(1-x)
    
    Parameters
    ----------
    s : bits
    p : probability of bits being 1
    
    Returns
    -------
    '''
    p = p.copy()
    p[p<eps]=eps
    p[p>1-eps]=1-eps
    s = np.int32(s)
    return np.sum(s*slog(p)+(1-s)*np.log1p(-p),axis=axis)

def lnPr_activation(s,a,axis=-1):
    '''
    Compute probability of bits s given Bernoulli probabilities p
    Assuming factorized distribution
    \prod p^x (1-p)^(1-x)
    
    Parameters
    ----------
    Returns
    -------
    '''
    return np.sum(s*a-np.log1p(sexp(a)),axis=axis)

def ground_state(a,eps=1e-12,axis=-1):
    '''
    Compute energy of zero vector given activatoins `a`
    
    Parameters
    ----------
    Returns
    -------
    '''
    return np.sum(np.log1p(sexp(a)),axis=axis)

def unique_counts(samples):
    '''
    
    Parameters
    ----------
    Returns
    -------
    '''
    samples = map(tuple,samples)
    counts = defaultdict(int)
    for s in samples:
        counts[s]+=1
    k,v = zip(*counts.items())
    return np.array(k),np.array(v)

def bernoulli_entropy(p,eps=1e-12):
    '''
    Entropy of Bernoulli distributed variable with probability `p`
    
    Parameters
    ----------
    p : numeric
        Values should lie between 0 and 1
        
    Returns
    -------
    '''
    p = p.copy()
    q = 1-p
    p[p<eps]=eps
    q[q<eps]=eps
    return -(p*np.log(p)+q*np.log(q))

def bernoulli_entropy_activation(a,eps=1e-12):
    '''
    
    Parameters
    ----------
    Returns
    -------
    '''
    p = f(a)
    q = 1-p
    q[q<eps]=eps
    return -(p*a+np.log(q))

def hashint64(x,N):
    '''
    Convert a bit vector to a float128
    Not to be confused with `pylab.packbits`
    
    Parameters
    ----------
    x : boolean or binary vector
    N : positive integer, number of bits in each vector
    Returns
    -------
    int64 : integer, stored in int64, whose binary bits match x
    '''
    if N>63:
        raise ValueError('No more than 63 bits can be safely stored in int64')
    return x.dot(2**np.int64(np.arange(N)))

def unhashint64(x,N):
    '''
    Unpack bits from number b; inverse of `hashint64()`
    Not to be confused with `pylab.unpackbits`
    
    Parameters
    ----------
    x : int64
        integer stored in int64, whose binary bits match x
    N : positive integer, number of bits in each vector
    
    Returns
    -------
    b : boolean or binary vector, unpacked
    '''
    if not x.dtype==np.int64:
        raise ValueError('Expected to unpack bit data from np.int64')
    if N>63:
        raise ValueError('No more than 63 bits can be safely stored in int64')
    x = x.copy()
    b = []
    for i in range(N):
        b.append(x%2)
        x >>= 1
    b = (np.uint8(b)==1)
    return b.T

def hashfloat128(x,N):
    '''
    Convert a bit vector to a float128
    Not to be confused with `pylab.packbits`
    
    Parameters
    ----------
    x : boolean or binary vector
    N : positive integer, number of bits in each vector
    Returns
    -------
    float128 : integer, stored in float128, whose binary bits match x
    '''
    x = (np.array(x)!=0)
    if not x.shape[-1]==N:
        raise ValueError('The last dimension of x should match the bit vector length N')
    if N>63:
        raise ValueError('No more than 63 bits are safe at the moment')
    return x.dot(2**np.float128(np.arange(N)))

def unhashfloat128(x,N):
    '''
    Unpack bits from number b; inverse of `hashbits()`
    Not to be confused with `pylab.unpackbits`
    
    Parameters
    ----------
    x : float128 
        integer stored in float128, whose binary bits match x
    N : positive integer, number of bits in each vector
    
    Returns
    -------
    b : boolean or binary vector, unpacked
    '''
    if not x.dtype==np.float128:
        raise ValueError('Expected to unpack bit data from np.float128')
    x = x.copy()
    b = []
    for i in range(N):
        b.append(x%2)
        x = np.floor(x*0.5)
    b = (np.uint8(b)==1)
    return b.T

def hashbits(x,N):
    '''
    Convert a bit vector to a float128
    Not to be confused with `pylab.packbits`
    
    Parameters
    ----------
    x : boolean or binary vector
    N : positive integer, number of bits in each vector
    Returns
    -------
    complex256 : complex256 encoding a pair of integers stored in float128, whose binary bits match x
    '''
    x = np.uint8(x)
    if not x.shape[-1]==N:
        raise ValueError('The last dimension of x should match the bit vector length N')
    if N>126:
        raise ValueError('No more than 126 bits can be stored this way')
    if N<=63:
        return np.complex256(hashfloat128(x,N))
    real = hashfloat128(x[...,:63],63)
    imag = hashfloat128(x[...,63:],N-63)
    return np.complex256(real+1j*imag)

def unhashbits(x,N):
    '''
    Unpack bits from number b; inverse of `hashbits()`
    Not to be confused with `pylab.unpackbits`
    
    Parameters
    ----------
    x : complex256 
        complex256 containing packed bits, produced by the `hashbits()` function
    N : positive integer, number of bits in each vector
    
    Returns
    -------
    b : boolean or binary vector, unpacked
    '''
    if not x.dtype==np.complex256:
        raise ValueError('Expected to unpack bit data from np.complex256')
    if N>126:
        raise ValueError('No more than 126 bits can be stored this way')
    x = x.copy()
    if N<=63:
        return unhashint64(np.int64(x.real),N)
    real = unhashint64(np.int64(x.real),63)
    imag = unhashint64(np.int64(x.imag),N-63)
    return np.concatenate([real,imag],axis=1)

def natent(p,eps=1e-12):
    '''
    
    Parameters
    ----------
    Returns
    -------
    '''
    # One nat is log2(e) bits
    p = np.array(p)
    p[p<eps]=eps
    p[p>1-eps]=1-eps
    return -(p*np.log(p)+(1-p)*np.log1p(-p))

def bitent(p,eps=1e-12):
    '''
    
    Parameters
    ----------
    Returns
    -------
    '''
    # One nat is log2(e) bits
    p = np.array(p)
    p[p<eps]=eps
    p[p>1-eps]=1-eps
    return -(p*np.log2(p)+(1-p)*np.log2(1-p))

