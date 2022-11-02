#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
pvalues.py

This module collects useful routines for 
working with p-values. 
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import statsmodels
import numpy as np
from numpy import random
from neurotools.util.array import find

def benjamini_hochberg_positive_correlations(pvalues,alpha):
    '''
    Derived from the following matlab code from Wilson Truccolo
    
        function [pID,pN] = fdr(p,q)
        % FORMAT pt = fdr(p,q)
        % 
        % p   - vector of p-values
        % q   - False Discovery Rate level
        %
        % pID - p-value threshold based on independence or positive dependence
        % pN  - Nonparametric p-value threshold
        %
        % This function takes a vector of p-values and a False Discovery Rate
        % (FDR). It returns two p-value thresholds, one based on an assumption of
        % independence or positive dependence, and one that makes no assumptions
        % about how the tests are correlated. For imaging data, an assumption of
        % positive dependence is reasonable, so it should be OK to use the first
        % (more sensitive) threshold.
        % 
        % Reference: Benjamini and Hochberg, J Royal Statistical Society. Series B
        % (Methodological), V 57, No. 1 (1995), pp. 289-300.
        % _____________________________________________________________________________
        % @(#)fdr.m 1.3 Tom Nichols 02/01/18
        % Wilson Truccolo: modified 10/19/2007

        p = sort(p(:));
        V = length(p);
        I = (1:V)';
        cVID = 1;
        cVN = sum(1./(1:V));
        pID = p(max(find(p<=I/V*q/cVID)));
        pN = p(max(find(p<=I/V*q/cVN)));

    Parameters
    ----------
    pvalues : list
        p-values to correct
    alpha : float in (0,1)
        target false-discovery rate

    Returns
    -------
    pID:
        p-value threshold based on independence or positive dependence
    pN:
        Nonparametric p-value threshold
    '''
    alpha = float(alpha)
    if alpha<=0 or alpha>=1: raise ValueError(
        'Desired false-discovery rate alpha should be '
        'between 0 and 1, got %s'%alpha)
    pvalues = sorted(np.ravel(np.array(list(pvalues))))
    V = len(pvalues)
    X = np.float64(np.arange(1,V+1))*alpha/V
    cVN = np.sum(1./np.arange(1,V+1))
    pID = np.where( pvalues<=X )[0]
    pID = pvalues[pID[-1]] if len(pID)>0 else 0#pvalues[0]
    pN  = np.where( pvalues<=X/cVN )[0]
    pN  = pvalues[pN [-1]] if len(pN )>0 else 0#pvalues[0]
    return pID, pN

def correct_pvalues_positive_dependent(pvalue_dictionary,verbose=0,alpha=0.05):
    '''
    Benjamini-Hochberg multiple-comparison correction
    assuming positive dependence.
    
    Parameters
    ----------
    pvalue_dictionary : dict 
        `label -> pvalue`
    
    Returns
    -------
    dict:
        Benjamini-Hochberg corrected dictionary assuming  
        positive correlations, 
        entries as `label -> pvalue, reject`
    '''
    labels, pvals = zip(*pvalue_dictionary.items())
    p_threshold = np.max(
        benjamini_hochberg_positive_correlations(
            pvals,alpha))
    reject = np.array(pvals)<p_threshold
    if verbose:
        print(
            'BENJAMINI-HOCHBERG POSITIVE CORRELATIONS\n\t',
        '\n\t'.join(map(str,zip(labels,pvals,reject))))
    corrected = dict(zip(labels,zip(pvals,reject)))
    return corrected

def correct_pvalues(pvalue_dictionary,verbose=0,alpha=0.05):
    '''
    This corrects for multiple comparisons using the
    Benjamini-Hochberg procedure, using either the 
    variance for positive dependence or no dependence,
    whichever is more conservative. 
    
    Parameters
    ----------
    pvalue_dictionary : dict 
        `label -> pvalue`
        This may also simply be a list of p-values.
    
    Returns
    -------
    dict:
        Benjamini-Hochberg corrected dictionary 
        correlations, entries as `label -> pvalue, reject`
    '''
    try:
        labels, pvals = zip(*pvalue_dictionary.items())
        reject, pvals_corrected, alphacSidak, alphacBonf = \
          statsmodels.sandbox.stats.multicomp.multipletests(
            pvals, alpha=alpha, method='fdr_bh')
        if verbose:
            print('BENJAMINI-HOCHBERG\n\t','\n\t'.join(
                map(str,
                    zip(labels,pvals_corrected,reject))))
        corrected = dict(zip(labels,zip(pvals_corrected,reject)))
        return corrected
    except:
        # Version for array input
        pvals = np.float32(pvalue_dictionary)
        reject, pvals_corrected, alphacSidak, alphacBonf = \
          statsmodels.sandbox.stats.multicomp.multipletests(
            pvals, alpha=alpha, method='fdr_bh')
        return pvals_corrected, reject
    
    

def bootstrap_statistic_two_sided(
    statistic, 
    test_population, 
    null_population, 
    ntrials=1000):
    '''
    Estimate the probability that `statistic` is 
    significantly **larger** in `test_population` 
    compared to `null_population` using bootstrap 
    resampling. 
    
    Bootstrapped p-values addresses nonlinearity and
    non-Gaussian dispersion, it **is not** a cure for 
    limited data. The number of data points in the test
    and null populations should be sufficiently large. 
    I recommend no less than 20. 
    
    Parameters
    ----------
    statistic: function
        A function that accepts a resampled collection of
        data from the population and returns some 
        scalar statistic.
    test_population:
        Values in the test population.
    null_population:
        Values in the null population. 
    
    Other Parameters
    ----------------
    ntrials: positive int; default 1000
        Number of random samples to use
    
    Returns
    --------
    pvalue: float in [0,1]
        Bootstrapped p-value
    '''
    n = len(test_population)
    null     = statistic(null_population)
    observed = statistic(test_population)
    T = observed-null
    delta = abs(T)
    null_samples = array([
        statistic(random.choice(null_population,n)) 
        for i in xrange(ntrials)])
    null_delta   = abs(null_samples - null)    
    pvalue = mean(null_delta>delta)
    return pvalue


def bootstrap_median(
    test_population, null_population, ntrials=10000):
    '''
    Estimate pvalue for difference in medians using bootstrapping
    
    Parameters
    ----------
    test_population: list of samples
        Values in the test population.
    null_population: list of samples
        Values in the null population. 
    
    Other Parameters
    ----------------
    ntrials: positive int; default 1000
        Number of random samples to use
    
    Returns
    -------
    pvalue: float in [0,1]
        Bootstrapped p-value
    '''
    return bootstrap_statistic_two_sided(
        median, test_population, null_population)


def bootstrap_mean(
    test_population, null_population, ntrials=10000):
    '''
    Estimate pvalue for difference in means using 
    bootstrapping.
    
    Parameters
    ----------
    test_population: list of samples
        Values in the test population.
    null_population: list of samples
        Values in the null population. 
    
    Other Parameters
    ----------------
    ntrials: positive int; default 1000
        Number of random samples to use
    
    Returns
    -------
    pvalue: float in [0,1]
        Bootstrapped p-value
    '''
    return bootstrap_statistic_two_sided(means, test_population, null_population)


def bootstrap_compare_statistic_two_sided(
    statistic, popA, popB, ntrials=1000):
    '''
    Estimate pvalue using bootstrapping
    
    Parameters
    ----------
    popA: list of samples
        Values in population A.
    popB: list of samples
        Values in population B.
    statistic: function
        A function that accepts a resampled collection of
        data from the population and returns some 
        scalar statistic.
    
    Other Parameters
    ----------------
    ntrials: positive int; default 1000
        Number of random samples to use
    
    Returns
    -------
    delta: float
        Mean difference  of statistic(A)-statistics(B)
    pvalue: float in [0,1]
        Bootstrapped p-value
    '''
    nA = len(popA)
    nB = len(popB)
    n  = nA+nB
    allstats = concatenate([popA,popB])
    A = statistic(popA)
    B = statistic(popB)
    def sample():
        shuffle = random.permutation(allstats)
        draw_A, draw_B = shuffle[:nA],shuffle[nA:]
        s_a = statistic(draw_A)
        s_b = statistic(draw_B)
        return abs(s_a-s_b)
    null_samples = array([sample() for i in xrange(ntrials)])
    delta = abs(A-B)
    pvalue = mean(null_samples>delta)
    return delta,pvalue


def __sample_parallel_helper(params):
    '''
    Helper function used by
    `bootstrap_compare_statistic_two_sided_parallel()`.
    
    Parameters
    ----------
    params: netsted tuple
        Packed parameters `(jobID,arguments)`, where
        `arguments=(statistic, popA, popB, NA, NB, ntrials)`
    
    Returns
    -------
    i: int
        Job I
    result:
        Shuffled estimates of statistic(A)-statistic(B)
    '''
    (i,args) = params
    (statistic, popA, popB, NA, NB, ntrials) = args
    numpy.random.seed()
    if NA is None:
        NA = len(popA)
    elif not NA<=len(popA):
        raise ValueError((
            '# samples for group A overriden to %d is '
            'larger than number of elements in A (%d)'
        )%(NA,len(popA)))
    if NB is None:
        NB = len(popB)
    elif not NB<=len(popB):
        raise ValueError((
            '# samples for group B overriden to %d is '
            'larger than number of elements in B (%d)'
        )%(NB,len(popB)))
    result = []
    for i in range(ntrials):
        shuffle = random.permutation(
            concatenate([popA,popB]))
        result.append(
            abs(statistic(shuffle[:NA])-\
                statistic(shuffle[-NB:])))
    return i,result


def bootstrap_compare_statistic_two_sided_parallel(
    statistic, popA, popB, 
    NA=None, NB=None, ntrials=10000):
    '''
    Estimate pvalue using bootstrapping
    
    Parameters
    ----------
    statistic: function
        A function that accepts a resampled collection of
        data from the population and returns some 
        scalar statistic.
    popA: list of samples
        Values in population A.
    popB: list of samples
        Values in population B.
    
    Other Parameters
    ----------------
    NA: positive int; default len(A)
        Number of samples to draw from population A.
        This should be no larger than the size of A.
    NB: positive int; default len(B)
        Number of samples to draw from population B.
        This should be no larger than the size of B.
    ntrials: positive int; default 1000
        Number of random samples to use
    
    Returns
    -------
    delta: float
        Mean difference  of statistic(A)-statistics(B)
    pvalue: float in [0,1]
        Bootstrapped p-value
    
    '''
    jobs = array([
        (i,(statistic,popA,popB,NA,NB,100)) 
        for i in range(ntrials//100+1)])
    null_samples = array(list(
        flatten(parmap(__sample_parallel_helper,jobs))))
    A = statistic(popA)
    B = statistic(popB)    
    delta = abs(A-B)
    n = sum(null_samples>delta)
    pvalue = float(n+1)/(1+ntrials)
    return delta,pvalue


def bootstrap_compare_median(
    popA, popB, 
    NA=None, NB=None, ntrials=100000):
    '''
    Estimate pvalue for difference in medians using 
    bootstrapping.
    
    Parameters
    ----------
    popA: list of samples
        Values in population A.
    popB: list of samples
        Values in population B.
    
    Other Parameters
    ----------------
    NA: positive int; default len(A)
        Number of samples to draw from population A.
        This should be no larger than the size of A.
    NB: positive int; default len(B)
        Number of samples to draw from population B.
        This should be no larger than the size of B.
    ntrials: positive int; default 1000
        Number of random samples to use
    
    Returns
    -------
    delta: float
        Mean difference  of statistic(A)-statistics(B)
    pvalue: float in [0,1]
        Bootstrapped p-value
    '''
    return bootstrap_compare_statistic_two_sided_parallel(
        median, popA, popB, NA, NB, ntrials)


def bootstrap_compare_mean(
    popA, popB, 
    NA=None, NB=None, ntrials=100000):
    '''
    Estimate pvalue for difference in means using 
    bootstrapping.
    
    Parameters
    ----------
    popA: list of samples
        Values in population A.
    popB: list of samples
        Values in population B.
    
    Other Parameters
    ----------------
    NA: positive int; default len(A)
        Number of samples to draw from population A.
        This should be no larger than the size of A.
    NB: positive int; default len(B)
        Number of samples to draw from population B.
        This should be no larger than the size of B.
    ntrials: positive int; default 1000
        Number of random samples to use
    
    Returns
    -------
    delta: float
        Mean difference  of statistic(A)-statistics(B)
    pvalue: float in [0,1]
        Bootstrapped p-value
    '''
    return bootstrap_compare_statistic_two_sided_parallel(
        mean, popA, popB, NA, NB, ntrials)
    
    
