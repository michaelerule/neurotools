#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

"""
Routines for computing commonly used summary statistics not otherwise
available in pylab
"""

from matplotlib.mlab import find
import neurotools.stats.modefind as modefind
import numpy as np

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average  = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))

def crossvalidated_least_squares(a,b,K,regress=np.linalg.lstsq):
    '''
    predicts B from A in K-fold cross-validated blocks using linear
    least squares
    returns
        model coefficients x
        predicted values of b under crossvalidation
        correlation coefficient
        root mean squared error
    '''
    N = len(b)
    B = N/K
    x = {}
    predict = []
    for k in range(K):
        start = k*B
        stop  = start+B
        #if stop>N: stop = N
        if k>=K-1: stop = N
        trainB = append(b[:start  ],b[stop:  ])
        trainA = append(a[:start,:],a[stop:,:],axis=0)
        testB  = b[start:stop]
        testA  = a[start:stop,:]
        x[k] = regress(trainA,trainB)[0]
        reconstructed = dot(testA,x[k])
        error = np.mean((reconstructed-testB)**2)
        predict.extend(reconstructed)
    cc  = pearsonr(b,predict)[0]
    rms = np.sqrt(np.mean((np.array(b)-np.array(predict))**2))
    return x,predict,cc,rms

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
    ndarray
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
    
def pca(c,eps=1e-6):
    '''
    wrapper for w,v=eig(c)
    removes dimensions smaller than eps=1e-6 of the largest dimension
    sorts dimensions by descending weight
    '''
    w,v = np.linalg.eig(c)
    ok = w>eps*np.max(w)
    w=w[ok]
    v=v[:,ok]
    order = np.argsort(-w)
    w=w[order]
    v=v[:,order]
    return w,v
