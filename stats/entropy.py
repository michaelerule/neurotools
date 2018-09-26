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

'''
A couple entropy functions. 
These should be merged with the energy functions in the RBM library
'''

import numpy as np
from collections import defaultdict

def discrete_entropy_samples(samples):
    '''
    Entropy is in nats
    
    Parameters
    ----------
    samples : array-like
        1D array-like iterable of samples. Samples can be of any type, but
        must be hashable. 
    
    Returns
    -------
    float
        Shannon entropy of samples
    '''
    counts = defaultdict(int)
    for s in samples:
        counts[s]+=1
    return discrete_entropy_distribution(counts.values())

def discrete_entropy_distribution(x,minp=1e-19):
    '''
    Parameters
    ----------
    x : array like numeric
        List of frequencies or counts
    
    Returns
    -------
    float
        Shannon entropy of discrete distribution with observed `counts`
    
    '''
    x = np.array(x)
    p = x/np.sum(x)
    p[p<minp] = minp
    total = np.sum(p)
    return np.log(total)-np.sum(p*np.log(p))/total

def regularized_discrete_entropy(samples,N):
    '''
    Specify that samples come from a set of size N
    Some examples might not be observed
    Use a bound on their probability to compute entropy
    
    Maybe extrapolate the energy density?
    '''
    raise NotImplementedError("Not implemented")
    

from neurotools.stats.distributions import poisson_pdf
def poisson_entropy_nats(l):
    '''
    Approximate the entropy of a Poisson distribution in nats
    '''
    cutoff = int(np.ceil(l+4*np.sqrt(l)))+1
    p = poisson_pdf(arange(cutoff),l)
    return discrete_entropy_distribution(p)
