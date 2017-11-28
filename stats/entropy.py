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

def discrete_entropy_distribution(counts):
    '''
    Parameters
    ----------
    counts : iterable of integers
        List of frequency counts for discrete states
    
    Returns
    -------
    float
        Shannon entropy of discrete distribution with observed `counts`
    
    '''
    total = np.sum(counts)
    return np.sum(np.log(counts)*counts)/total - np.log(total)

def regularized_discrete_entropy(samples,N):
    '''
    Specify that samples come from a set of size N
    Some examples might not be observed
    Use a bound on their probability to compute entropy
    
    Maybe extrapolate the energy density?
    '''
    pass
