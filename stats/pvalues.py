#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

import statsmodels
import numpy as np
from numpy import random
from matplotlib.mlab import find

#TODO: fix imports
#from neurotools.jobs.parallel import *
#from numpy import *


def benjamini_hochberg_positive_correlations(pvalues,alpha):
    '''
    Derived from the following matlab code (c) Wilson Truccolo
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
    '''
    pvalues = sorted(ravel(pvalues))
    V = len(pvalues)
    X = float64(arange(1,V+1))*alpha/V
    cVN  = sum(1./arange(1,V+1))
    pID  = find( pvalues<=X )
    pID  = pvalues[pID[-1]] if len(pID)>0 else 0#pvalues[0]
    pN   = find( pvalues<=X/cVN )
    pN   = pvalues[pN [-1]] if len(pN )>0 else 0#pvalues[0]
    print(pID, pN)
    return pID, pN

def correct_pvalues_positive_dependent(pvalue_dictionary,verbose=0):
    '''
    Parameters
    ----------
    pvalue_dictionary : dict 
        `label -> pvalue`
    
    Returns
    -------
    dict:
        Benjamini-Hochberg corrected dictionary assuming positive 
        correlations, entries as `label -> pvalue, reject`
    '''
    labels, pvals = zip(*pvalue_dictionary.iteritems())
    p_threshold = max(*benjamini_hochberg_positive_correlations(pvals,0.05))
    reject = array(pvals)<p_threshold
    if verbose:
        print('BENJAMINI-HOCHBERG POSITIVE CORRELATIONS\n\t','\n\t'.join(map(str,zip(labels,pvals,reject))))
    corrected = dict(zip(labels,zip(pvals,reject)))
    return corrected


def correct_pvalues(pvalue_dictionary,verbose=0):
    '''
    Parameters
    ----------
    pvalue_dictionary : dict 
        `label -> pvalue`
    
    Returns
    -------
    dict:
        Benjamini-Hochberg corrected dictionary 
        correlations, entries as `label -> pvalue, reject`
    '''
    labels, pvals = zip(*pvalue_dictionary.iteritems())
    reject, pvals_corrected, alphacSidak, alphacBonf = \
      statsmodels.sandbox.stats.multicomp.multipletests(pvals, alpha=0.05, method='fdr_bh')
    if verbose:
        print('BENJAMINI-HOCHBERG\n\t','\n\t'.join(map(str,zip(labels,pvals_corrected,reject))))
    corrected = dict(zip(labels,zip(pvals_corrected,reject)))
    return corrected


def bootstrap_statistic_two_sided(statistic, test_population, null_population, ntrials=1000):
    '''
    Estimate pvalue using bootstrapping
    
    Parameters
    ----------
    
    Resturns
    --------
    '''
    n = len(test_population)
    null     = statistic(null_population)
    observed = statistic(test_population)
    T = observed-null
    delta = abs(T)
    null_samples = array([statistic(random.choice(null_population,n)) for i in xrange(ntrials)])
    null_delta   = abs(null_samples - null)    
    pvalue = mean(null_delta>delta)
    return pvalue


def bootstrap_median(test_population, null_population, ntrials=10000):
    '''
    Estimate pvalue for difference in medians using bootstrapping
    
    Parameters
    ----------
    
    Resturns
    --------
    '''
    return bootstrap_statistic_two_sided(median, test_population, null_population)


def bootstrap_mean(test_population, null_population, ntrials=10000):
    '''
    Estimate pvalue for difference in means using bootstrapping
    
    Parameters
    ----------
    
    Resturns
    --------
    '''
    return bootstrap_statistic_two_sided(means, test_population, null_population)


def bootstrap_compare_statistic_two_sided(statistic, population_A, population_B, ntrials=1000):
    '''
    Estimate pvalue using bootstrapping
    
    Parameters
    ----------
    
    Resturns
    --------
    '''
    nA = len(population_A)
    nB = len(population_B)
    n  = nA+nB
    allstats = concatenate([population_A,population_B])
    A = statistic(population_A)
    B = statistic(population_B)
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


def sample_parallel_helper(params):
    '''
    
    Parameters
    ----------
    params: (i,(statistic, population_A, population_B, NA, NB, ntrials))
    Resturns
    --------
    '''
    (i,(statistic, population_A, population_B, NA, NB, ntrials)) = params
    numpy.random.seed()
    if NA is None:
        NA = len(population_A)
    else:
        assert NA<=len(population_A)
    if NB is None:
        NB = len(population_B)
    else:
        assert NB<=len(population_B)
    result = []
    for i in range(ntrials):
        shuffle = random.permutation(concatenate([population_A,population_B]))
        result.append(abs(statistic(shuffle[:NA])-statistic(shuffle[-NB:])))
    return i,result


def bootstrap_compare_statistic_two_sided_parallel(statistic, population_A, population_B, NA=None, NB=None, ntrials=10000):
    '''
    Estimate pvalue using bootstrapping
    
    Parameters
    ----------
    
    Resturns
    --------
    '''
    problems     = array([(i,(statistic,population_A,population_B,NA,NB,100)) for i in range(ntrials//100+1)])
    null_samples = array(list(flatten(parmap(sample_parallel_helper,problems))))
    A = statistic(population_A)
    B = statistic(population_B)    
    delta = abs(A-B)
    n = sum(null_samples>delta)
    pvalue = float(n+1)/(1+ntrials)
    return delta,pvalue


def bootstrap_compare_median(population_A, population_B, NA=None, NB=None, ntrials=100000):
    '''
    Estimate pvalue for difference in medians using bootstrapping
    
    Parameters
    ----------
    
    Resturns
    --------
    '''
    return bootstrap_compare_statistic_two_sided_parallel(median, population_A, population_B, NA, NB, ntrials)


def bootstrap_compare_mean(population_A, population_B, NA=None, NB=None, ntrials=100000):
    '''
    Estimate pvalue for difference in means using bootstrapping
    
    Parameters
    ----------
    
    Resturns
    --------
    '''
    return bootstrap_compare_statistic_two_sided_parallel(mean, population_A, population_B, NA, NB, ntrials)
    
    

    
    
    
    
    
