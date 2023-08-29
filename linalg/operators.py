#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Functions for generating discrete representations of 
certain operators, either as matrices or in terms of 
their discrete Fourier coefficients. 
'''

from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import numpy.linalg
import scipy
from neurotools.util.functions import sexp

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
    of length ``N``, with circularly-wrapped boundary condition.

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
        Size of operator, or width if ``H`` is provided
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
    
    This is uses an approximately radially-symmetric 
    Laplacian in a 3×3 neighborhood.
    
    If a ``mask`` is provided, this supports a
    ``'neumann'`` boundary condition, which amounts to 
    clamping the derivative to zero at the boundary, 
    and a ``'dirichlet'`` boundary condition,
    which amounts to clamping the values to zero at
    the boundary. 
    
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
        mask = np.array(mask)>0
        if not mask.shape==(nrows,ncols):
            raise ValueError(('Mask with shape %s provided, '
            'but operator shape is %s')%(mask.shape,(W,H)))
    
    boundary = str(boundary).lower()[0]
    if not boundary in 'dn':
        raise ValueError('Boundary can be "dirichlet" or "neumann"')
        
    Ad = adjacency2D(W,H,circular)
        
    if boundary == 'n':
        # We get a mirrored boundary if we remove neighbors first
        Ad[~mask.ravel(),:] = Ad[:,~mask.ravel()] = 0
        if not mask is None:
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
    of length ``N``

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
        one-dimensional domain of lenght ``N``
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

def circular_derivative_operator(N):
    '''
    Circulat discete differentiation in the frequeincy
    domain.
    

    Parameters
    ----------
    N : int
        Size of the domain    
    
    Returns
    -------
    result: N×N np.float32
        Fourier transform of the circular discrete 
        derivative.
    '''
    delta = np.zeros((N,))
    delta[0]=-1
    delta[-1]=1
    x = np.fft.fft(delta)
    return bp.flot32(x)

def truncated_derivative_operator(N):
    '''
    Discrete derivative operator, projecting from N to 
    N-1 dimensions (spatial domain). 
    
    Parameters
    ----------
    N : int
        Size of the domain    
    
    Returns
    -------
    result: (N-1)×N np.float32
        Matrix D such that Dx = Δx
    '''
    return eye(N-1,N,1) - eye(N-1,N,0)

def terminated_derivative_operator(N):
    '''
    Discrete derivative, using {-1,1} at endpoints and
    ½{-1,0,1} in the interior. 
    
    This operator will have two zero eigenvalues for ``N``
    even and three zero eigenvalues for ``N`` odd. 
    
    Parameters
    ----------
    N : int
        Size of the domain    
    
    Returns
    -------
    result: N×N np.float32
        Matrix D such that Dx = Δx
    '''
    D = eye(11,11,1) - eye(11,11,-1)
    D = np0.array(D)
    D[1:-1]*=0.5
    D[0,0] = -1
    D[-1,-1]=1
    return D

def pad1up(N):
    '''
    Interpolation operator going from N to N+1 samples.
    Used to re-sample the discrete derivative to preserve
    dimension. 
    
    Parameters
    ----------
    N : int
        Size of the domain    
    
    Returns
    -------
    result: N×(N-1) np.float32
    '''
    K = N-1
    interpoints = linspace(0,K-1,N)
    ii = int32(floor(interpoints))
    ii,ff = np0.divmod(interpoints,1)
    ii = np.int32(ii)
    D = np0.zeros((N,N-1),dtype=np0.float32)
    D[arange(N-1),ii[:-1]]=1.-ff[:-1]
    D[arange(N-1),ii[:-1]+1]=ff[:-1]
    D[-1,-1]=1.
    return D

def spaced_derivative_operator(N):
    '''
    Discrete derivative, upsampled via linear interpolation
    to preserve dimension. 
    
    This will have 1 zero eigenvalue for odd N and two
    zero eigenvalues for even N. 
    
    Parameters
    ----------
    N : int
        Size of the domain    
    
    Returns
    -------
    result: N×N np.float32
    '''
    return pad1up(N)@truncated_derivative(N)
    
    


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

def gaussian2DblurOperator(n,sigma,normalize='left'):
    '''
    Returns a 2D Gaussan blur operator for a n × n domain.
    Constructed as a tensor product of two 1d blurs.
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
    # normalize rows so density is conserved
    # Stochastic matrix 
    # on right: columns sum to 1; 
    # on left: rows sum to 1
    if normalize=='left':    
        # Rows sum to 1 for left operator
        op/=sum(op,0)[None,:]
    elif normalize=='right':
        # Columns sum to 1 for right operator
        op/=sum(op,1)[:,None]
    elif normalize!=None:
        raise ValueError('normalize must be "left", "right", or None')
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
    
    
    
    
    
