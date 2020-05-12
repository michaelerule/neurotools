#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

"""
Routines for computing commonly used summary statistics not otherwise
available in pylab
"""

import numpy as np
import scipy
import random
try:
    from  matplotlib.mlab import find
except:
    def find(x):
        return np.where(np.array(x).ravel())[0]

from   scipy.stats.stats import describe

def nrmse(estimate,true,axis=None):
    '''
    Normalized root mean-squared error.
    Parameters
    ----------
    estimate : array-like
        Estimated data values
    true: array-like
        True data values
    '''
    X1,X2 = estimate,true
    v1 = np.var(X1,axis=axis)
    v2 = np.var(X2,axis=axis)
    normalize = v2**-0.5#(v2*v2)**-0.25
    return np.mean((X1-X2)**2,axis=axis)**0.5*normalize

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average  = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))

import neurotools.tools

def partition_trials_for_crossvalidation(x,K,shuffle=False):
    '''
    Split trial data into crossvalidation blocks.
    
    Parameters
    ----------
    x : list
        List of trial data to partition. Each entry in the list should
        be a Ntimepoints x Ndatapoints array.
    K : int
        Number of crossvalidation blocks to compute
    
    Returns
    -------
    spans : list
        List of trial indecies to use for each block
    '''
    # Number of trials
    N    = len(x)
    if K>N:
        raise ValueError('#xval-blocks K=%d>#trials N=%d'%(K,N))
    # Length of each trial
    lens = np.array([xi.shape[0] for xi in x])
    # Total length of the data
    L    = np.sum(lens)
    # Average target length of each block
    B    = L/K
    if shuffle:
        # Randomly re-order the trials
        order    = np.random.permutation(N)
        lens     = lens[order]
        indecies = order
    else:
        # Use trials in order
        indecies = np.arange(N)
    # Handle this one as a special case
    if K==N:
        # leave-one-out crossvalidation
        return list([np.array([i]) for i in indecies])
    # Cumulative length of data so far including length of current trial
    C    = np.cumsum(lens)
    # Target total cumulative length of each block
    Bk   = (np.arange(K)+1)*B
    # Edges between blocks: try to keep things close to the desired
    # length. 
    edge = np.argmin(np.abs(C[:,None]-Bk[None,:]),axis=0)
    edge = np.array(sorted(list(set(edge))))
    if len(edge)!=K:
        # Sometimes this will return duplicate edges, which is
        # a bit of a bug. Solution? 
        # Don't use length-based partitions in this case
        edge = np.int32(0.5+np.linspace(0,N,K+2)[1:-1])
    edge = np.array(sorted(list(set(edge))))
    # Get star, end point for each edge, to define blocks to keep
    a      = np.concatenate([[0],edge[:-1]+1])
    b      = np.concatenate([edge[:-1]+1,[N]])
    result = [indecies[np.arange(ai,bi)] for ai,bi in zip(a,b)]
    result = [r for r in result if len(r)>0]
    if len(result)!=K:
        raise ValueError('Could not divide N=%d trials into K=%d blocks'%(N,K))
    return result
    
def polar_error(x,xh,units='degrees',mode='L1'):
    '''
    Compute error for polar measurements, 
    wrapping the circular variable appropriately.

    Parameters
    ----------
    x: array-like
        true valies (in degrees)
    hx: array-like
        estimated values (in degrees)

    Other Parameters
    ----------------
    units: str, default "degrees"
        Polar units to use. Either "radians" or "degrees"
    mode: str, default 'L1'
        Error method to use. Either 'L1' (mean absolute error) or 
        'L2' (root mean-squared error)

    Returns
    -------
    err:
        Circularly-wrapped error
    '''
    x  = np.array(x).ravel()
    xh = np.array(xh).ravel()
    e  = np.abs(x-xh)
    k  = {'radians':np.pi,'degrees':180}[units]
    e[e>k] = 2*k-e[e>k]
    if mode=='L1':
        return np.mean(np.abs(e))
    if mode=='L2':
        return np.mean(np.abs(e)**2)**.5
    raise ValueError('Mode should be either "L1" or "L2"')

error_functions = {
    'correlation':lambda x,xh: scipy.stats.stats.pearsonr(x,xh)[0],
    'L2'         :lambda x,xh: np.mean((x-xh)**2)**0.5,
    'L1'         :lambda x,xh: np.mean(np.abs(x-xh)),
    'L1_degrees' :lambda x,xh: polar_error(x,xh,mode='L1',units='degrees'),
    'L2_degrees' :lambda x,xh: polar_error(x,xh,mode='L2',units='degrees'),
    'L1_radians' :lambda x,xh: polar_error(x,xh,mode='L1',units='radians'),
    'L2_radians' :lambda x,xh: polar_error(x,xh,mode='L2',units='radians')
}

import warnings
def add_constant(data):
    data = np.array(data)
    if not len(data.shape)==2:
        raise ValueError('Expected a Nsamples x Nfeatures 2D array')
    Nsamples,Nfeatures = data.shape
    #if Nsamples<Nfeatures:
    #    warnings.warn("data shape is %dx%d\n# samples < # features; is data transposed?"%data.shape)
    return np.concatenate([data,np.ones((data.shape[0],1))],axis=1)

def trial_crossvalidated_least_squares(a,b,K,
    regress=None,
    reg=1e-10,
    shuffle=False,
    errmethod='L2',
    **kwargs):
    '''
    predicts B from A in K-fold cross-validated blocks using linear
    least squares. I.e. find w such that B = Aw

    Parameters
    ----------
    a : array
        List of trials for independent variables; For every trial, 
        First dimension should be time or number of samples, etc. 
    b : vector
        List of trials for dependent variables
    K : int
        Number of cross-validation blocks

    Other Parameters
    ----------------
    regress : function, optional
        Regression function, defaults to `np.linalg.lstsq`
        (if providing another function, please match the 
        call signature of `np.linalg.lstsq`)
    reg : scalar, default 1e-10
        L2 regularization penalty
    shuffle : bool, default False
        Whether to shuffle trials before crossvalidation
    errmethod: String
        Method used to compute the error. Can be 'L1' (mean absolute error)
        'L2' (root mean-squared error) or 'correlation' (pearson correlation
        coefficient). 
    add_constant: bool, default False
        Whether to append an additional constand offset feature to the
        data. The returned weight matrix will have one extra entry, at the
        end, reflecting the offset, if this is set to True. 
    
    Returns
    -------
    w, array-like:
        model coefficients x from each cross-validation
    bhat, array-like:
        predicted values of b under crossvalidation
    error :
        root mean squared error from each crossvalidation run
    '''

    if not errmethod in error_functions:
        raise ValueError('Error method should be one of '+\
                         ', '.join(error_functions.keys()))
    
    # Check shape of data
    a = np.array([np.array(ai) for ai in a])
    b = np.array([np.array(bi) for bi in b])
    Ntrial = len(a)
    if K<=1:
        raise ValueError('# crossvalidation blocks (K) should be >1')
    if Ntrial<K:
        raise ValueError('Expected more than K trials to use!')
    if len(b)!=Ntrial:
        raise ValueError('X and Y should have same # of trials')
    Nsampl = sum([ai.shape[0] for ai in a])
    
    # Get typical block length
    B = Nsampl//K 
    
    # Determine trial groups for cross-validation
    groups = partition_trials_for_crossvalidation(a,K,shuffle=shuffle)
    if len(groups)!=K:
        raise ValueError('Expected K groups for crossvalidation!')
    
    # Define regression solver if none provided
    if regress is None:
        def regress(A,B):
            Q = A.T.dot(A) + np.eye(A.shape[1])*reg*A.shape[0]
            return np.linalg.solve(Q, A.T.dot(B))
    
    if 'add_constant' in kwargs and kwargs['add_constant']:
        a = np.array([add_constant(ai) for ai in a])
    
    # Iterate over each cross-validation
    x    = {}
    Bhat = {}
    for k in range(K):
        train  = np.concatenate(
            groups[1:]       if k==0 
            else groups[:-1] if k==K-1 
            else groups[:k]+groups[k+1:])
        trainA = np.concatenate(a[train])
        trainB = np.concatenate(b[train])
        x[k]   = regress(trainA,trainB)
        # Trials might be shuffled, so we predict them one-by-one
        # Then assign them to their correct slot to preserve original order.
        for i in groups[k]:
            Bhat[i] = a[i].dot(x[k])

    # Convert dictionaries to list
    Bhat = np.array([Bhat[i] for i in range(Ntrial)])
    x    = np.array([x   [i] for i in range(K)     ])

    # Apply error function within each crossvalidation block
    efn  = error_functions[errmethod]
    errs = [efn(np.concatenate(b[g]),np.concatenate(Bhat[g])) for g in groups]

    return x,np.concatenate(Bhat),errs

def partition_data_for_crossvalidation(a,b,K):
    '''
    For predicting B from A, partition both training and testing
    data into K-fold cross-validation blocks.

    Parameters
    ----------
    a : array
        Independent variables; First dimension should be time
        or number of samples, etc. 
    b : vector
        dependent variables
    K : int
        Number of cross-validation blocks
    
    Returns
    -------
    trainA : list
        list of training blocks for independent variables A
    trainB : list
        list of training blocks for dependent variables B
    testA : list
        list of testing blocks for independent variables A
    testB : list
        list of testing blocks for dependent variables B
    '''
    # Check shape of data
    a   = np.array(a)
    b   = np.array(b)
    N,h = np.shape(a)
    if N<h: raise ValueError('1st axis of `a` must be time. is `a` transposed?')
    # Get typical block length
    B = N//K
    trainA, trainB, testA, testB = [],[],[],[]
    # Iterate over each cross-validation
    for k in range(K):
        # Start and stop of testing data range
        start = k*B
        stop  = start+B if k<K-1 else N
        # Training data (exclude testing block)
        trainB.append(np.append(b[:start,...],b[stop:,...],axis=0))
        trainA.append(np.append(a[:start,:  ],a[stop:,:  ],axis=0))
        # Testing data
        testB.append(b[start:stop,...])
        testA.append(a[start:stop,:  ])
    return trainA,trainB,testA,testB

def crossvalidated_least_squares(a,b,K,regress=None,reg=1e-10,blockshuffle=None):
    '''
    predicts B from A in K-fold cross-validated blocks using linear
    least squares. I.e. find w such that B = Aw

    Parameters
    ----------
    a : array
        Independent variables; First dimension should be time
        or number of samples, etc. 
    b : vector
        dependent variables
    K : int
        Number of cross-validation blocks

    Other Parameters
    ----------------
    regress : function, optional
        Regression function, defaults to `np.linalg.lstsq`
        (if providing another function, please match the 
        call signature of `np.linalg.lstsq`)
    reg : scalar, default 1e-10
        L2 regularization penalty
    blockshuffle : positive int or None, default None
        If not None, should be a positive integeter indicating the 
        block-size in which to shuffle the input data before
        breaking it into cross-validation blocks.
    
    Returns
    -------
    w, array-like:
        model coefficients x from each cross-validation
    bhat, array-like:
        predicted values of b under crossvalidation
    cc, number:
        correlation coefficient
    rms, number:
        root mean squared error
    '''
    # Check shape of data
    a = np.array(a)
    b = np.array(b)
    N,h = np.shape(a)
    if N<h: 
        raise ValueError('1st axis of `a` must be time. is `a` transposed?')

    B = N//K # Get typical block length

    # Optionally shuffle data
    if not blockshuffle is None:
        blockshuffle = int(blockshuffle)
        if blockshuffle>B//2:
            raise ValueError('Shuffle block len should be <½ xvalidation block len')
            ab = block_shuffle(concatenate([aa,bb],axis=1),500)
            a = ab[:,:h]
            b = ab[:,h:]

    x = {}
    predict = []
    if regress is None:
        #def regress(trainA,trainB):
        #    return np.linalg.lstsq(trainA,trainB,rcond=None)[0]
        def regress(A,B):
            Q = A.T.dot(A) + np.eye(A.shape[1])*reg*A.shape[0]
            return np.linalg.solve(Q, A.T.dot(B))
            
    # Iterate over each cross-validation
    for k in range(K):
        # Start and stop of testing data range
        start = k*B
        stop  = start+B
        if k>=K-1: stop = N
        # Training data (exclude testing block)
        trainB = np.append(b[:start,...],b[stop:,...],axis=0)
        trainA = np.append(a[:start,:  ],a[stop:,:  ],axis=0)
        # Testing data
        testB  = b[start:stop,...]
        testA  = a[start:stop,:  ]
        # Train regression model
        x[k] = regress(trainA,trainB)
        reconstructed = np.dot(testA,x[k])
        predict.extend(reconstructed)
    predict = np.array(predict)
    
    # Correlation coefficient
    if len(np.shape(b))==1:
        cc = scipy.stats.stats.pearsonr(b,predict)[0]
    else:
        cc = [scipy.stats.stats.pearsonr(bi,pi)[0] for bi,pi in zip(b.T,predict.T)]

    # Root mean-squared error
    rms = np.sqrt(np.mean((np.array(b)-np.array(predict))**2))
    return x,np.array(predict),cc,rms

def print_stats(g,name='',prefix=''):
    '''
    computes, prints, and returns
    mean
    median
    minimum
    maximum
    '''
    #mode = modefind.modefind(g,0)
    mn   = np.mean(g)
    md   = np.median(g)
    mi   = np.min(g)
    mx   = np.max(g)
    #print(prefix,'mode    %s\t%0.4f'%(name,mode))
    print(prefix,'mean    %s\t%0.4f'%(name,mn))
    print(prefix,'median  %s\t%0.4f'%(name,md))
    print(prefix,'minimum %s\t%0.4f'%(name,mi))
    print(prefix,'maximum %s\t%0.4f'%(name,mx))
    return mn,md,mi,mx

def outliers(x,percent=10,side='both'):
    '''
    Reject outliers from data based on percentiles.

    Parameters
    ----------
    x : ndarary
        1D numeric array of data values
    percent : number
        percent between 0 and 100 to remove
    side : str
        'left' 'right' or 'both'. Default is 'both'. Remove extreme
        values from the left / right / both sides of the data
        distribution. If both, the percent is halved and removed
        from both the left and the right
    Returns
    -------
    ndarray:
        Boolean array of same shape as x indicating outliers
    '''
    N = len(x)
    remove = np.zeros(len(x),'bool')
    if   side=='left':
         remove |= x<np.percentile(x,percent)
    elif side=='right':
         remove |= x>np.percentile(x,100-percent)
    elif side=='both':
         remove |= x<np.percentile(x,percent*0.5)
         remove |= x>np.percentile(x,100-percent*0.5)
    else:
        raise ValueError('side must be left, right, or both')
    return remove

def reject_outliers(x,percent=10,side='both'):
    '''
    Reject outliers from data based on percentiles.

    Parameters
    ----------
    x : ndarary
        1D numeric array of data values
    percent : number
        percent between 0 and 100 to remove
    side : str
        'left' 'right' or 'both'. Default is 'both'. Remove extreme
        values from the left / right / both sides of the data
        distribution. If both, the percent is halved and removed
        from both the left and the right
    Returns
    -------
    ndarray
        Values with outliers removed
    kept
        Indecies of values kept
    removed
        Indecies of values removed
    '''
    N = len(x)
    remove = outliers(x,percent,side)
    to_remove = find(remove==True)
    to_keep   = find(remove==False)
    return x[to_keep], to_keep, to_remove
    
def pca(x,n_keep=None):
    '''
    w,v = pca(x,n_keep=None)
    Performs PCA on data x, keeping the first n_keep dimensions
    
    Parameters
    ----------
    x: ndarray
        Nsamples x Nfeatures array on which to perform PCA
    n_keep : int
        Number of principle components to retain
        
    Returns
    -------
    w : weights (eigenvalues)
    v : eigenvector (principal components)
    '''
    assert x.shape[1]<=x.shape[0]
    cov = x.T.dot(x)
    w,v = scipy.linalg.eig(cov)
    o   = np.argsort(-w)
    w,v = w[o].real,v[:,o].real
    if n_keep is None: n_keep = len(w)
    w,v = w[:n_keep],v[:,:n_keep]
    return w,v

def covariance(x,y=None,sample_deficient=False,reg=0.0,centered=True):
    '''
    Covariance matrix for `Nsamples` x `Nfeatures` matrix.
    Data are *not* centered before computing covariance.
    
 
    Parameters
    ----------
    x : Nsamples x Nfeatures array-like
        Array of input features
        
    Other parameters
    ----------------
    y : Nsamples x Nyfeatures array-like
        Array of input features
    sample_deficient: bool, default False
        Whether the data contains fewer samples than it does features. 
        If False (the default), routine will raise a `ValueError`.
    reg: positive scalar, default 0
        Diagonal regularization to add to the covariance
    centered: boolean, default True
        Whether to subtract the means from the data before taking the
        covariace.
    
    Returns
    -------
    C : np.array
        Sample covariance matrix
    '''
    x = np.array(x)
    Nsamples,Nfeatures = x.shape
    if not sample_deficient and Nfeatures>Nsamples:
        raise ValueError('x should be Nsample x Nfeature where Nsamples >= Nfeatures');
    if centered:
        x = x - np.mean(x,axis=0)[None,:]

    # Covariance of x
    if y is None:
        #if np.all(np.isfinite(x)): 
        C = x.T.dot(x)/Nsamples
        #else:
        #    C = np.zeros((Nfeatures,Nfeatures))
        #    for i in range(Nfeatures):
        #        C[i,i+1:] = np.nanmean(x[:,i:i+1]*x[:,i+1:],axis=0)
        #    C = C+C.T
        #    for i in range(Nfeatures):
        #        C[i,i]    = np.nanvar (x[:,i])
        R = np.eye(Nfeatures)*reg
        return C+R
    
    # Cross-covariance between x and y
    y = np.array(y)
    if len(y.shape)==1:
        y = np.array([y]).T
    Nysamples,Nyfeatures = y.shape
    if not Nysamples==Nsamples:
        raise ValueError('1st dimension of x and y (# of samples) should be the same')
    if not abs(reg)<1e-12:
        raise ValueError('Cross-covariance does not support non-zero regularization')
    if centered:
        y = y - np.mean(y,axis=0)[None,:]
    
    C = x.T.dot(y)/Nsamples
    return C
    
            

class description:
    '''
    quick statistical description
    
    TODO: move this to stats
    '''
    def __init__(self,data):
        '''
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        self.N, (self.min, self.max),self.mean,self.variance,self.skewness,self.kurtosis = describe(data)
        self.median = np.median(data)
        self.std  = np.std(data)

        # quartiles
        self.q1   = np.percentile(data,25)
        self.q3   = self.median
        self.q2   = np.percentile(data,75)

        # percentiles
        self.p01  = np.percentile(data,1)
        self.p025 = np.percentile(data,2.5)
        self.p05  = np.percentile(data,5)
        self.p10  = np.percentile(data,10)
        self.p90  = np.percentile(data,90)
        self.p95  = np.percentile(data,95)
        self.p975 = np.percentile(data,97.5)
        self.p99  = np.percentile(data,99)

    def __str__(self):
        '''
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        result = ''
        for stat,value in self.__dict__.iteritems():
            result += ' %s=%0.2f '%(stat,value)
        return result

    def short(self):
        '''
        Abbreviated statistical summary
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        abbreviations = {
            'N':'N',
            'min':'mn',
            'max':'mx',
            'mean':u'μ',
            'variance':u'σ²',
            'skewness':'Sk',
            'kurtosis':'K'
        }
        result = []
        for stat,value in self.__dict__.iteritems():
            if stat in abbreviations:
                result.append('%s:%s '%(abbreviations[stat],shortscientific(value)))
        return ' '.join(result)
