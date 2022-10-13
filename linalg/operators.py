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


from neurotools.functions import sexp

'''
Functions for generating discrete representations of certain operators. 
Note: This is partially redundant with `neurotools.spatial.kernels`.
'''


def adjacency1D(L,circular=True):
    '''
    1D adjacency matrix.
    
    Parameters
    ----------
    N: int
        Size of operator
    
    Returns
    -------
    L: NxN sparse array
        Adjacency matrix.
    '''
    # Make adjacency matrix
    # [1 0 1 .. 0]
    A1d = scipy.sparse.eye(L,k=-1) + scipy.sparse.eye(L,k=1)
    if circular:
        A1d += scipy.sparse.eye(L,k=L-1) + scipy.sparse.eye(L,k=1-L) 
    return A1d


def laplacian1D_circular(N):
    '''
    Laplacian operator on a closed, discrete, one-dimensional domain
    of length `N`, with circularly-wrapped boundary condition.

    Parameters
    ----------
    N: int
        Size of operator
    
    Returns
    -------
    L: NxN array
        Matrix representation of Laplacian operator on a finite, 
        discrete domain of length N.
    '''
    return adjacency1D_circular(N)-2*np.eye(N)

    
def adjacency2D(L,H=None,circular=True):
    '''
    2D adjacency matrix in 3x3 neighborhood.
    
    Parameters
    ----------
    L: int
        Size of operator, or width if H is provided
    H: int
        Height of operator, if different from width
    circular: bool
        If true, operator will wrap around the edge in both directions
    
    Returns
    -------
    L: NxN sparse array
        Adjacency matrix.
    '''
    W = int(L)
    H = W if H is None else int(H)
    a1drow = adjacency1D(H,circular=circular)
    a1dcol = adjacency1D(W,circular=circular)
    A2d_cross  = scipy.sparse.kronsum(a1drow,a1dcol)
    A2d_corner = scipy.sparse.kron   (a1dcol,a1drow)
    A2d = (A2d_cross*2/3+A2d_corner*1/3).toarray()
    return A2d


def laplacian2D(L,H=None,circular=True,mask=None,boundary='dirichlet'):
    '''
    Build a discrete Laplacian operator.
    This is uses an approximatelly radially-symmetric Laplacian in a 3×3 cell
    
    Parameters
    ----------
    L: int
        Size of operator, or width if H is provided
    H: int
        Height of operator, if different from width
    circular: bool
        If true, operator will wrap around the edge in both directions
    mask: LxL np.bool
        Which pixels are in the domain
    boundary: str
        Can be 'dirichlet', which is a zero boundary, or
        'neumann', which is a reflecting boundary.
        
    Returns
    -------
    scipy.sparse.csr_matrix
    '''
    
    W = int(L)
    H = W if H is None else int(H)
    
    ncols = H
    nrows = W
    
    if not mask is None:
        mask = np.array(mask)
        if not mask.shape==(nrows,ncols):
            raise ValueError(('Mask with shape %s provided, '
            'but operator shape is %s')%(mask.shape,(W,H)))
    
    boundary = str(boundary).lower()[0]
    if not boundary in 'dn':
        raise ValueError('Boundary can be "dirichlet" or "neumann"')
        
    Ad = adjacency2D(W,H,circular)
        
    if boundary == 'n':
        # We get a reflecting boundary if we remove neighbors first
        if mask is None:
            Ad[~mask.ravel(),:] = Ad[:,~mask.ravel()] = 0
        Lp = Ad - np.diag(sum(Ad,0))
    elif boundary =='d':
        # We get a zero boundary if we remove neighbors after
        Lp = Ad - np.diag(sum(Ad,0))
        if not mask is None:
            Lp[~mask.ravel(),:] = Lp[:,~mask.ravel()] = 0
    return scipy.sparse.csr_matrix(Lp)

def adjacency2D_circular(N):
    '''
    Adjacency matric on a closed NxN domain with circularly-wrapped boundary.

    Parameters
    ----------
    N: int
        Size of operator
    
    Returns
    -------
    L: N²xN² array
        Adjacency matrix. The diagonal is zero (no self edges)
    '''
    A1d = adjacency1D_circular(N)
    return scipy.sparse.kronsum(A1d,A1d).toarray()

def adjacency2d_rotational(L):
    '''
    2D adjacency matrix with nonzero weights for corner neighbors to improve
    rotational symmetry. 
    '''
    A1d = eye(L,k=-1) + eye(L,k=1) + eye(L,k=L-1) + eye(L,k=1-L) 
    A2d_cross  = scipy.sparse.kronsum(A1d,A1d).toarray()
    A2d_corner = scipy.sparse.kron(A1d,A1d).toarray()
    return A2d_cross*2/3+A2d_corner*1/3

def laplacian1D(N):
    '''
    Laplacian operator on a closed, discrete, one-dimensional domain
    of length `N`

    Parameters
    ----------
    N: int
        Size of operator
    
    Returns
    -------
    L: NxN array
        Matrix representation of Laplacian operator on a finite, 
        discrete domain of length N.
    '''
    L = -2*np.eye(N)+np.eye(N,k=1)+np.eye(N,k=-1)
    L[0,0] = L[-1,-1] = -1
    return L

def laplacianFT1D(N):
    '''
    Parameters
    ----------
    N : int
        Size of the domain    
    
    Returns
    -------
    x : np.array
        Fourier transform of discrete Laplacian operator on a discrete
        one-dimensional domain of lenght `N`
    '''
    precision = np.zeros((N,))
    if (N%2==1):
        precision[N//2]=2
        precision[N//2+1]=precision[N//2-1]=-1
    else:
        precision[N//2+1]=precision[N//2]=1
        precision[N//2+2]=precision[N//2-1]=-1
    x = np.fft.fft(precision)
    return x

def wienerFT1D(N):
    '''
    Square-root covariance operator for standard 1D Wiener process
    
    Parameters
    ----------
    N : size of operator
    
    Returns
    -------
    '''
    x = laplacianop(N)
    x[np.abs(x)<1e-5]=1
    sqrtcov = 1/np.sqrt(x)
    return sqrtcov

def diffuseFT1D(N,sigma):
    '''
    Fourier transform of a Gaussian smoothing kernel

    Parameters
    ----------
    N : int
        Size of the domain    
    
    Returns
    -------
    '''
    kernel = np.exp(-.5/(sigma**2)*(np.arange(N)-N//2)**2)
    kernel /= np.sum(kernel)
    kernel = np.roll(kernel,N//2)
    return np.fft.fft(kernel).real

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
    x = np.fft.fft(delta)
    return x

def differentiator(N):
    '''
    Fourier space discrete differentiaion

    Parameters
    ----------
    N : int
        Size of the domain    
    
    Returns
    -------
    '''
    delta = np.zeros((N,))
    delta[0]=-1
    delta[-1]=1
    x = np.fft.fft(delta)
    return x

def integrator(N):
    '''
    Parameters
    ----------
    N : int
        Size of the domain    
    
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

def gaussian1DblurOperator(n,sigma,truncate=1e-5,normalize=True):
    '''
    Returns a 1D Gaussan blur operator of size n
    
    Parameters
    ----------
    n: int
        Length of buffer to apply blur
    sigma: positive number
        Standard deviation of blur kernel
    
    Other Parameters
    ----------------
    truncate: positive number, defaults to 1e-5
        Entries in the operator smaller than this (relative to the largest value)
        will be rounded down to zero. 
    '''
    x   = np.linspace(0,n-1,n); # 1D domain
    tau = 1.0/sigma**2;       # precision
    k   = sexp(-tau*x**2);    # compute (un-normalized) 1D kernel
    op  = scipy.linalg.special_matrices.toeplitz(k,k);     # convert to an operator from n -> n
    # normalize rows so density is conserved
    if normalize:
        op = op/np.sum(op,1)[:,None]
    # truncate small entries
    big = np.max(op)
    toosmall = truncate*big
    op[op<toosmall] = 0
    # normalize rows again so density is conserved
    if normalize:
        op = op/np.sum(op,1)[:,None]
    return op

def circular1DblurOperator(n,sigma,truncate=1e-5):
    '''
    Returns a circular 1D Gaussan blur operator of size n
    
    Parameters
    ----------
    n: int
        Length of circular buffer to apply blur
    sigma: positive number
        Standard deviation of blur kernel
    
    Other Parameters
    ----------------
    truncate: positive number, defaults to 1e-5
        Entries in the operator smaller than this (relative to the largest value)
        will be rounded down to zero. 
    '''
    x   = np.linspace(0,n-1,n); # 1D domain
    tau = 1.0/sigma**2;       # precision
    k   = sexp(-tau*(x-n/2.0)**2);    # compute (un-normalized) 1D kernel
    op  = np.array([np.roll(k,i+n//2) for i in range(n)])
    # normalize rows so density is conserved
    op /= np.sum(op,1)
    # truncate small entries so things stay sparse
    big = np.max(op)
    toosmall = truncate*big
    op[op<toosmall] = 0
    # normalize rows so density is conserved
    op /= np.sum(op,1)
    return op

def separable_guassian_blur(op,x):
    n = len(op)
    return op @ x @ ou.T

def gaussian2DblurOperator(n,sigma):
    '''
    Returns a 2D Gaussan blur operator for a n x n sized domain
    Constructed as a tensor product of two 1d blurs of size n
    '''
    x   = np.linspace(0,n-1,n) # 1D domain
    tau = 1.0/sigma**2       # precision
    k   = sexp(-tau*x**2)    # compute (un-normalized) 1D kernel
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
    x = np.float64(abs(x))/2.0*np.pi
    return np.piecewise(x,[x<=np.pi],[lambda x:(np.cos(x)+1)/4.0])

def log_cosine_basis(N=range(1,6),t=np.arange(100),base=2,offset=1,normalize=True):
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
    kernels = np.array([cosine_kernel(s-k) for k in N]) 
    # evenly spaced in log-time
    if normalize:
        kernels = kernels/np.log(base)/(offset+t) 
    # correction for change of variables, kernels integrate to 1 now
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
    B = log_cosine_basis(arange(N),t,base=b,offset=0)
    return B
    
    
    
    
    
