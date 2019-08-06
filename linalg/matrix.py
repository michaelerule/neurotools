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
#from neurotools.system import *
#if sys.version_info<(3,):
#    from itertools import imap as map
# END PYTHON 2/3 COMPATIBILITY BOILERPLATE

'''
Collecting matrix-related subroutines
'''

import numpy as np
import scipy
import scipy.linalg
import numpy
import numpy.linalg

chol = scipy.linalg.cholesky

def triu_elements(M,k=0):
    '''
    Somewhat like matlab's "diag" function, but for lower triangular matrices

    Pack N*(N-1) elements into the upper triangle of an NxN Matrix
    or
    Return the N*(N-1) elements from the upper triangle as an NxN matrix

    triu_elements(randn(D*(D+1)//2))
    '''
    if len(M.shape)==2:
        # M is a matrix
        if not M.shape[0]==M.shape[1]:
            raise ValueError("Extracting upper triangle elements supported only on square arrays")
        # Extract upper trianglular elements
        i = np.triu_indices(M.shape[0],k=k)
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

def tril_elements(M,k=0):
    '''
    Somewhat like matlab's "diag" function, but for lower triangular matrices

    Pack N*(N-1) elements into the lower triangle of an NxN Matrix
    or
    Return the N*(N-1) elements from the lower triangle as an NxN matrix
    
    tril_elements(randn(D*(D+1)//2))
    '''
    if len(M.shape)==2:
        # M is a matrix
        if not M.shape[0]==M.shape[1]:
            raise ValueError("Extracting upper triangle elements supported only on square arrays")
        # Extract upper trianglular elements
        i = np.tril_indices(M.shape[0],k=k)
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
    x = np.array(x).ravel()
    return x.reshape((x.shape[0],1))
    
def row(x):
    '''
    Ensure that x is a row vector
    '''
    x = np.array(x).ravel()
    return x.reshape((1,x.shape[0]))

def rcond(x):
    '''
    Reciprocal condition number
    '''
    return 1./np.linalg.cond(x)

def check_finite_real(M):
    '''
    Check that all entries in array M are finite and real-valued
    '''
    if np.any(~np.isreal(M)):
        raise ValueError("Complex value encountered for real vector")
    if np.any(~np.isfinite(M)):
        raise ValueError("Non-finite number encountered")

# need a faster covariance matrix checker
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
    C = np.real(C)
    if np.any(~np.isfinite(C)):
        raise ValueError("Covariance matrix contains NaN or ±inf!")
    if not np.all(np.abs(C-C.T)<eps):
        raise ValueError("Covariance matrix is not symmetric up to precision %0.1e"%eps)
    
    # Get just highest eigenvalue
    maxe = np.real(scipy.linalg.decomp.eigh(C,eigvals=(N-1,N-1))[0][0])
    
    # Get just lowest eigenvalue
    mine = np.real(scipy.linalg.decomp.eigh(C,eigvals=(0,0))[0][0])

    #if np.any(w<-eps):
    #    raise ValueError('Covariance matrix contains eigenvalue %0.3e<%0.3e'%(np.min(w),-eps)) 
    if mine<0:
        raise ValueError('Covariance matrix contains negative eigenvalue %0.3e'%mine) 
    if (mine<eps):
        C = C + np.eye(N)*(eps-mine)
    # trucate spectrum at some small value
    # w[w<eps]=eps
    # Very large eigenvalues can also cause numeric problems
    # w[w>1./eps]=1./eps;
    # maxe = np.max(np.abs(w))
    # if maxe>10./eps:
    #     raise ValueError('Covariance matrix eigenvalues %0.2e is larger than %0.2e'%(maxe,10./eps))
    # Rebuild matrix
    # C = v.dot(np.diag(w)).dot(v.T)
    # Ensure symmetry (only occurs as a numerical error for very large matrices?)
    C = 0.5*(C+C.T)
    return C

# need a faster covariance matrix checker
def check_covmat_fast(C,N=None,eps=1e-6):
    '''
    Verify that matrix M is a size NxN precision or covariance matirx
    '''
    if not type(C)==np.ndarray and len(C.shape)==2:
        raise ValueError("Covariance matrix should be a 2D numpy array")
    if N is None: 
        N = C.shape[0]
    if not C.shape==(N,N):
        raise ValueError("Expected size %d x %d matrix"%(N,N))
    if np.any(~np.isreal(C)):
        raise ValueError("Covariance matrices should not contain complex numbers")
    C = np.real(C)
    if np.any(~np.isfinite(C)):
        raise ValueError("Covariance matrix contains NaN or ±inf!")
    if not np.all(np.abs(C-C.T)<eps):
        raise ValueError("Covariance matrix is not symmetric up to precision %0.1e"%eps)
    try:
        ch = chol(C)
    except numpy.linalg.linalg.LinAlgError:
        # Check smallest eigenvalue if cholesky fails
        mine = np.real(scipy.linalg.decomp.eigh(C,eigvals=(0,0))[0][0])
        if np.any(mine<-eps):
            raise ValueError('Covariance matrix contains eigenvalue %0.3e<%0.3e'%(mine,-eps)) 
        if (mine<eps):
            C = C + np.eye(N)*(eps-mine)
    C = 0.5*(C+C.T)
    return C

def real_eig(M,eps=1e-9):
    '''
    This code expects a real hermetian matrix
    and should throw a ValueError if not.
    This is redundant to the scipy eigh function (use that one instead!)
    
    Returns
    -------
    w : eigenvalues
    v : eigenvectors
    '''
    if not (type(M)==np.ndarray):
        raise ValueError("Expected array; type is %s"%type(M))
    if np.any(np.abs(np.imag(M))>eps):
        raise ValueError("Matrix has imaginary entries >%0.2e; will not extract real eigenvalues"%eps)
    M = np.real(M)
    w,v = np.linalg.eig(M)
    if np.any(abs(np.imag(w))>eps):
        raise ValueError('Eigenvalues with imaginary part >%0.2e; matrix has complex eigenvalues'%eps)
    w = np.real(w)
    order = np.argsort(abs(w))[::-1]
    w = w[order]
    v = v[:,order]
    return w,v.real

def psd_eig(M,eps=1e-9):
    '''
    This code expects a real hermetian matrix
    and should throw a ValueError if not.
    This is probably redundant to the scipy eigh function.
    Do not use.
    '''
    e,v = real_eig(M,eps)
    o   = np.argsort(-np.abs(e))
    e   = e[o]
    v   = v[:,o]
    if np.any(e<-eps):
        raise ValueError('Matrix has negative eigenvalues')
    e = np.maximum(e,eps)
    return e,v

def logdet(C,eps=1e-6,safe=0):
    '''
    Logarithm of the determinant of a matrix
    Works only with real-valued positive definite matrices
    '''
    try:
        return 2.0*np.sum(np.log(np.diag(chol(C))))
    except numpy.linalg.linalg.LinAlgError:
        if safe: C = check_covmat(C,eps=eps)
        w = np.linalg.eigh(C)[0]
        w = np.real(w)
        w[w<eps]=eps
        det = np.sum(np.log(w))
        return det

def solt(a,b):
    '''
    wraps solve_triangular
    automatically detects lower vs. upper triangular
    '''
    if np.allclose(a, scipy.linalg.special_matrices.tril(a)): # check if lower triangular
        return scipy.linalg.solve_triangular(a,b,lower=1)
    if np.allclose(a, scipy.linalg.special_matrices.triu(a)): # check if upper triangular
        return scipy.linalg.solve_triangular(a,b,lower=0)
    raise ValueError('a matrix is not triangular')

def rsolt(a,b):
    '''
    wraps solve_triangular, right hand solution
    solves system x A = B for triangular A
    '''
    return solt(b.T,a.T).T

def rsolve(a,b):
    '''
    wraps solve, applies to right hand solution
    solves system x A = B
    '''
    return scipy.linalg.solve(b.T,a.T).T

def qf(A,S=None):
    '''
    Matrix quatratic forms A*S*A.T
    If S is none, compute A*A.T
    '''
    if S is None: return A.dot(A.T)
    return A.dot(S).dot(A.T)

def abserr(M1,M2):
    '''
    Mean absolute element-wise difference between teo matrices
    '''
    norm = 0.5*np.mean(np.abs(M1))+0.5*np.mean(np.abs(M2))
    err  = np.mean(np.abs(M1-M2))
    return err/norm

def errmx(stuff):
    '''
    Takes a list of objects and prints out a matirx of the pairwise element-wise mean absolute differences.
    All objects mst have the same shape.
    '''
    RMSE = np.zeros((len(stuff),len(stuff)))
    for i in range(len(stuff)):
        for k in range(len(stuff)):
            RMSE[i,k] = abserr(stuff[i],stuff[k])
    print('Errors:')
    print('\n'.join(['\t'+'   '.join(['%7.3f %%'%(n*100) for n in row]) for row in RMSE]))
    
def cholupdate(R,x):
    '''
    Rank-1 update to a cholesky factorization
    Possibly slower than simply recomputing
    
    Test
    q  = randn(10,10)
    qq = q.T.dot(q)
    ch = chol(qq)
    x  = randn(10)
    xx = outer(x,x)
    pp = qq+xx
    cp = chol(pp)
    c2 = cholupdate(ch,x.T)
    print(abserr(c2,cp))
    '''
    p = np.size(x)
    x = x.T
    for k in range(p):
        r = np.sqrt(R[k,k]**2 + x[k]**2)
        c = r/R[k,k]
        s = x[k]/R[k,k]
        R[k,k] = r
        R[k,k+1:p] = (R[k,k+1:p] + s*x[k+1:p])/c
        x[k+1:p]= c*x[k+1:p] - s*R[k, k+1:p]
    return R

def cholupdate_eye(R):
    '''
    Idenetity matrix update to a cholesky factorization
    Probably slower than simply recomputing
    
    Test
    q  = randn(10,10)
    qq = q.T.dot(q)
    ch = chol(qq)
    pp = qq+eye(10)
    cp = chol(pp)
    c2 = cholupdate_eye(ch)
    print(abserr(c2,cp))
    '''
    n = R.shape[0]
    for i in range(n):
        q = np.zeros(n)
        q[i]=1
        R = cholupdate(R,q)
    return R
    
    
def cartesian_product(*arrays):
    '''
    https://stackoverflow.com/questions/11144513/
    numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    '''
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def _getAplus(A):
    '''
    Please see the documentation for nearPDHigham
    '''
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    '''
    Please see the documentation for nearPDHigham
    '''
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    '''
    Please see the documentation for nearPDHigham
    '''
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPDHigham(A, nit=10):
    '''
    Computes a positive definite matrix "close" to matrix $X$ using 
    the algorithm of Higham (2000). 
    
    This is based on a 
    [Stackoverflow answer](https://stackoverflow.com/a/10940283/900749),
    and relevant intellectual property rights are retained by user
    [sega sai](https://stackoverflow.com/users/1269140/sega-sai), 
    [Stackoverflow](https://stackoverflow.com), or Higham (2000).
    
    Parameters
    ----------
    X : np.array
        Square, real-valued matrix 

    Returns
    -------
    np.array
        Positive semi-definite matrix close to $X$
    '''
    n = A.shape[0]
    W = np.identity(n) 
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return np.array(Yk)

def nearPSDRebonatoJackel(A,epsilon=1e-10):
    '''
    Computes a positive definite matrix "close" to matrix $X$ using 
    the algorithm of Rebonato and Jackel (1999).
    
    This is based on a 
    [Stackoverflow answer](https://stackoverflow.com/a/18542094/900749)
    
    Parameters
    ----------
    X : np.array
        Square, real-valued matrix 
    epsilon : non-negative scalar, default 1e-10
        minimum eigenvalue

    Returns
    -------
    np.array
        Positive semi-definite matrix close to $X$
    '''
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval,epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec,vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    return B*B.T

def cinv(X,repair=False):
    '''
    Invert positive matrix $X$ using cholesky factorization. The function
    `numpy.linalg.cholesky` is aliased as `chol` in this library, in 
    analogy to matlab. `chol` returns a upper-triangular matrix such
    that $L = \operatorname{chol}(X)$ and $X = L^T L$. The inverse of
    $X$ is $X^{-1} = (L^T L)^{-1} = L^{-1} L^{-T}$. 
    
    This routine uses [LAPACK dtrtri](http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga97c5ddb103f5f6bc2dc060886aaf2ffc.html#ga97c5ddb103f5f6bc2dc060886aaf2ffc)
    . See also [scipy.linalg.lapack.dtrtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dtrtri.html#scipy.linalg.lapack.dtrtri).
        
    Parameters
    ----------
    X : np.array
        Square, real-valued, symmetric positive definite matirx
    repair : boolean, default is False
        If true, and if $X$ is not positive definite, then this routine
        will attempt to operate on a new positive matrix close to $X$
    Returns
    -------
    matrix :
        The inverse of $X$ computed using Cholesky factorization
    '''
    try:
        ch = chol(X)
    except numpy.linalg.LinAlgError:
        if repair:
            X = nearPSDRebonatoJackel(0.5*(X+X.T));
            try:
                ch = chol(X)
            except numpy.linalg.LinAlgError:
                raise ValueError("Repairing matrix to positive definite failed")         
        else:
            raise ValueError("Matrix is not positive-definite within machine precision")
    # Should have lower-triangular factor ch
    # X = ch.T.dot(ch)
    # X^-1 = ch^-1 ch^-T
    ich,info = scipy.linalg.lapack.dtrtri(ch)
    if info!=0:
        if info<0:
            raise ValueError('lapack.dtrtri encountered illegal argument in position %d'%-info)
        else:
            raise ValueError('lapack.dtrtri encountered zero diagonal element at %d'%info)
    return ich.dot(ich.T)

def wheremax(a):
    '''
    Returns the indecies of the maximum element in a multi-dimensional 
    array.
    
    Parameters
    ----------
    a : np.array
        Numpy multi-dimensional array
        
    Returns
    -------
    tuple : 
        Tuple of indecies indicating the maximum element
    '''
    return np.unravel_index(a.argmax(), a.shape)

def wheremin(a):
    '''
    Returns the indecies of the minimum element in a multi-dimensional 
    array.
    
    Parameters
    ----------
    a : np.array
        Numpy multi-dimensional array
        
    Returns
    -------
    tuple : 
        Tuple of indecies indicating the minimum element
    '''
    return np.unravel_index(a.argmin(), a.shape)
    
def reglstsq(A, B, reg=1e-15):
    '''
    Regularized least squares. 
    Solves Ax=B for x with L2 regularization
    
    Parameters
    ----------
    A : two-dimensional numpy array
        Matrix of observations of the explanatory/independent variables
    B : two-dimensional numpy array
        Matrix of observations of the response/dependent variables
    
    Other Parameters
    ----------------
    reg : positive float
        Small positive L2 regularization, default is 1e-15
        
    Returns
    -------
    w : weight vector   
    '''
    Q = A.T.dot(A) + np.eye(A.shape[1])*reg*A.shape[0]
    return np.linalg.solve(Q, A.T.dot(B))

def Ldistance(X,M,L=2,eps=1e-3):
    '''
    L-n norm distance between two vectors
    '''
    X = np.abs(X-M)
    if L==0:
        thr = np.median(X)*eps
        return -np.sum(X>thr,axis=1)
    return -np.sum(X**L,axis=1)**(1/L)

def Llasso(X,M,L=2,eps=1e-3):
    '''
    Lasso-like distance (I think? need to check this one)
    '''
    X = np.abs(X-M)
    if L==0:
        thr = np.median(X)*eps
        return -np.sum(X>thr,axis=1)
    return -np.sqrt(np.sum(X**2,axis=1))**L

def rmatrix(h):
    '''
    Generate 2D rotation matrix for angle `h`
    
    Parameters
    ----------
    h : float
        Angle of rotation in radians
        
    Returns
    -------
    M : 2x2 numpy array
        2D rotation matrix for `h` radians.
    '''
    ch,sh = np.cos(h),np.sin(h)
    return np.array([[ch,sh],[-sh,ch]])

def ldiv(A,B):
    '''
    Solve AX=B for X = A^{-1}B
    i.e. find matrix X which when right-multiplied with A is close to B
    '''
    #return scipy.linalg.lstsq(A,B)[0]
    return np.linalg.solve(A,B)

def rdiv(A,B):
    '''
    Solves for X=AB^{-1} , i.e. BX = A
    '''
    return ldiv(B.T,A.T).T

def autopredict(z,reg=1e-9):
    '''
    Solve mutual prediction problem for 
    T x N covariates z
    Approximates z = z Q, where diag(Q) is zero
    '''
    T,N  = z.shape
    Sz   = z.T @ z *T**-2 + np.eye(N,N)*reg
    idxs = np.arange(N)
    Q    = np.zeros((N,N))
    for i in idxs:
        j = sorted(list(set(idxs)-set([i])))
        Q[i,j] = np.linalg.solve(Sz[j,:][:,j], Sz[j,:][:,i])
    return Q.T

from scipy.linalg.special_matrices import kron
def kronsum(A,B):
    '''
    Kronecker sum
    '''
    A = np.array(A)
    B = np.array(B)
    m = B.shape[0]
    n = A.shape[0]
    return kron(A,np.eye(m)) + kron(np.eye(n),B)

def top_v(C):
    '''
    Return the leading eigenvector of a covariance matrix
    '''
    N = C.shape[0]
    e,v = scipy.linalg.eigh(C,eigvals=(N-1,N-1))
    return v.squeeze()

def normedcovariance(C):
    '''
    Divide a covariance matrix by its largest eigenvalue
    '''
    N = C.shape[0]
    e,v = scipy.linalg.eigh(C,eigvals=(N-1,N-1))
    return C/e

def selector_matrix(b):
    '''
    Create matrix that extracts a subset of elements.
    Returns selector matrix S such that:
    
    X[b] = S @ X
    
    Parameters
    ----------
    b : boolean vector
    '''
    b = np.array(b)
    N = len(b)
    K = np.sum(b)
    S = np.zeros((K,N))
    S[arange(K),neurotools.tools.find(b)]=1
    return S
