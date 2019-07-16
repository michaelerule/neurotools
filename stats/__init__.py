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
        error = np.mean((reconstructed-testB)**2)
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

def covariance(x,sample_deficient=False,reg=0.0):
    '''
    Covariance matrix for `Nsamples` x `Nfeatures` matrix.
    Data are *not* centered before computing covariance.

    Parameters
    ----------
    
    Returns
    -------
    '''
    a,b = x.shape
    if not sample_deficient and b>a:
        raise ValueError('x should be Nsample x Nfeature where Nsamples >= Nfeatures');
    C = x.T.dot(x)/x.shape[0]
    C = C + np.eye(C.shape[0])*reg
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
