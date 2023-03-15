#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Additional hypothesis-testing routines to supplement
``scipy.stats`` and ``statsmodels``.
"""
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
import scipy.stats
from typing import NamedTuple 



from neurotools.stats.information import betapr
def two_tailed_sampled_test(delta):
    '''
    Directly test whether a population ``delta``
    is above or below zero more than chance. 
    This is for use
    with bootstrap/shuffle tests when stronger assumptions
    may be inaccurate or risk false-positives.
    '''
    delta = np.float32(delta).ravel()
    # This is very under-powered?
    if len(delta)<100:
        raise RuntimeError(
            'You probably don\'t want to use this '
            'with fewer than 100 samples.')
    k0 = sum(delta<0)
    k1 = sum(delta>0)
    if max(k0,k1)<5:
        raise RuntimeError(
            'There probably aren\'t enough samples')
    pr1gtr0 = betapr(k0,len(delta))
    pr0gtr1 = betapr(sum(delta>0),len(delta))
    pvalue  = 1-(1-min(pr1gtr0, pr0gtr1))**2
    return pvalue


class ZTestResult(NamedTuple):
    z: float
    pvalue: float
def ztest_from_moments(μ1,s1,μ2,s2):
    '''
    Calculate z-test given moments from two samples. 
    '''
    Δ = μ1 - μ2
    S = np.sqrt(s1**2 + s2**2)
    z = Δ/S
    pvalue = 2*scipy.stats.norm.sf(abs(z))
    return ZTestResult(z,pvalue)


import neurotools.stats
class WeightedTtestResult(NamedTuple):
    t:float
    pvalue:float
    dof:float
    alternative:str
    sem:float
    mu:float
    s:float
def weighted_ttest_1samp(
    x,
    w,
    alternative='two-sided'):
    '''
    Test if mean of independent samples ``x``
    with weights ``w`` is different from zero using
    a to-tailed one-sample t-test. 
    '''
    n   = np.sum(w) # Effective sample size
    dof = n-1       # Degrees of freedom
    μ,σ = neurotools.stats.weighted_avg_and_std(x,w)
    s   = σ*np.sqrt(n/(n-1)) # sample s.d.
    sem = s/np.sqrt(n)       # standard error
    t = μ/sem                # score
    if alternative=='two-sided':
        pvalue = scipy.stats.t(dof).sf(np.abs(t))*2
    if alternative=='greater':
        pvalue = scipy.stats.t(dof).sf(t)
    if alternative=='less':
        pvalue = scipy.stats.t(dof).cdf(t)
    return WeightedTtestResult(t,pvalue,dof,alternative,sem,μ,s)



def beta_propotion_test(
    a1,b1,
    a2,b2,
    npts = 1000,
    eps  = 0.5):
    '''
    (experimental)
    
    Use a Beta distribution model to determine whether
    two propotions are significantly different. 
    
    Parameters
    ----------
    a1: positive int
        Number of items in category 0, group 1
    b1: positive int
        Number of items in category 1, group 1
    a2: positive int
        Number of items in category 0, group 2
    b2: positive int
        Number of items in category 1, group 2
        
    Returns
    -------
    mudelta: float
        When positive: The probability of being in the 
        second category is larger for the second group, 
        compared to the first. 
    p: float
        Two-tailed p-value for significant diffierence
        in rates between the groups. 
    '''
    from scipy.stats import beta

    # Model count data using beta distribution
    d1 = beta(eps+a1, eps+b1)
    d2 = beta(eps+a2, eps+b2)

    # Integrate to get distr. of differences
    ll = np.linspace(0,1,npts+1)
    pdelta = np.convolve(d1.pdf(ll), d2.pdf(ll)[::-1])
    pdelta /= np.sum(pdelta)

    # average difference
    l2 = np.linspace(-1,1,2*npts+1)
    mudelta = l2@pdelta

    # P value
    if mudelta>0:
        p = np.sum(pdelta[l2<0])*2
    else:
        p = np.sum(pdelta[l2>0])*2
    return mudelta,p


def cohen_d(x,y):
    '''
    Calculate Cohen's d effect-size summary for independent
    samples from two unpaired populations.
    '''
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    vx = np.nanvar(x, ddof=1)
    vy = np.nanvar(y, ddof=1)
    v  = ((nx-1)*vx + (ny-1)*vy) / dof
    d = (np.nanmean(x) - np.nanmean(y)) / np.sqrt(v)
    # Cohen, Jacob (1988).
    # Sawilowsky, S (2009)
    bins = [-np.inf,0.01,0.20,0.50,0.80,1.20,2.0,np.inf]
    sizenames = ['Very small',
        'Small',
        'Medium',
        'Large',
        'Very large',
        'Huge']
    i = np.digitize(abs(d),bins,right=True)-1
    return d, sizenames[i]
