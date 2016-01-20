
from neurotools.modefind import *
import numpy
from numpy.linalg import lstsq

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average  = numpy.average(values, weights=weights)
    variance = numpy.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, math.sqrt(variance))

weighted_avg_std = weighted_avg_and_std
weighted_mean_std = weighted_avg_and_std

def crossvalidated_least_squares(a,b,K,regress=lstsq):
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
        error = mean((reconstructed-testB)**2)
        predict.extend(reconstructed)
        # print 'block',k
    cc  = pearsonr(b,predict)[0]
    rms = sqrt(mean((array(b)-array(predict))**2))
    return x,predict,cc,rms


def print_stats(g,name='',prefix=''):
    '''
    computes, prints, and returns
    mode
    mean
    median
    '''
    mode = modefind(g,0)
    mn   = mean(g)
    md   = median(g)
    print prefix,'mode    %s\t%0.4f'%(name,mode)
    print prefix,'mean    %s\t%0.4f'%(name,mn)
    print prefix,'median  %s\t%0.4f'%(name,md)
    return mode,mn,md








