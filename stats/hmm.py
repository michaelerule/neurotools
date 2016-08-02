#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
'''
Routines concerning discrete time hidden markov models
'''

import numpy as np
from neurotools.functions import slog
from neurotools.stats.distributions import poisson_logpdf

def hmm_poisson_parameter_guess(X,Y,N):
    '''
    Based on a sequence of inferred states X, estimate the log of
    the transition matrix A, the prior vector P, and the state
    probability matrix B.

    Parameters
    ----------
    X : ndarary
        1D integer array of hidden states with values in 0..N-1
    Y : ndarary
        1D integer array of observations
    N : positive integer
        Number of states
    '''
    # Estimate model parameters from best-guess X
    p1  = np.mean(X)
    p01 = np.sum(np.diff(X)== 1)/(1+np.sum(X==0))
    p10 = np.sum(np.diff(X)==-1)/(1+np.sum(X==1))
    mu0 = np.mean(Y[X==0])
    mu1 = np.mean(Y[X==1])
    params = (p1,p01,p10,mu0,mu1)

    # Prior for state at first observation
    logP = np.array([np.log1p(-p1),slog(p1)])
    # State transition array
    logA = np.array([
        [np.log1p(-p01), slog(p01)],
        [slog(p10), np.log1p(-p10)]],
        dtype=np.float64)
    # Poisson process rates
    O = np.arange(N)      # List of available states
    logB = np.array([
        poisson_logpdf(O, mu0),
        poisson_logpdf(O, mu1)])
    return logP,logA,logB,params

def hmm_viterbi(Y,logP,logA,logB):
    '''
    See https://en.wikipedia.org/wiki/Viterbi_algorithm

    Parameters
    ----------
    Y : 1D array
        Observations (integer states)
    logP : array shape = (nStates ,)
        1D array of priors for initial state
        given in log probability
    logA : array (nStates,nStates)
        State transition matrix given in log probability
    logB : ndarray K x N
        conditional probability matrix
        log probabilty of each observation given each state
    '''
    K = len(logP)         # Number of states
    T = len(Y)            # Number of observations
    N = np.shape(logB)[1] # Number of states
    Y = np.int32(Y)

    assert np.shape(logA)==(K,K)
    assert np.shape(logB)==(K,N)

    # The initial guess for the first state is initialized as the
    # probability of observing the first observation given said 
    # state, multiplied by the prior for that state.
    logT1 = np.zeros((K,T),'float') # Store probability of most likely path
    logT1[:,0] = logP + logB[:,Y[0]]

    # Store estimated most likely path
    T2 = np.zeros((K,T),'float')

    # iterate over all observations from left to right
    for i in range(1,T):
        # iterate over states 1..K (or 0..K-1 with zero-indexing)
        for s in range(K):
            # The likelihood of a new state is the likelihood of 
            # transitioning from either of the previous states.
            # We incorporate a multiplication by the prior here
            log_filtered_likelihood = logT1[:,i-1] + logA[:,s] + logB[s,Y[i]]
            best = np.argmax(log_filtered_likelihood)
            logT1[s,i] = log_filtered_likelihood[best]
            # We save which state was the most likely
            T2[s,i] = best

    # At the end, choose the most likely state, then
    # Iterate backwards over the data and fill in the state estimate
    X     = np.zeros((T,) ,'int'  ) # Store our inferred hidden states
    X[-1] = np.argmax(logT1[:,-1])
    for i in range(1,T)[::-1]:
        X[i-1] = T2[X[i],i]
    return X

def hasNaN(x):
    '''
    Faster way to test if array contains NaN
    '''
    return np.isnan(np.sum(x))

def hmm_poisson_baum_welch(Y,initial=None):
    '''
    Fit Hidden Markov Model using Expectation Maximization. For now
    it is limited to two latent states.

    Parameters
    ----------
    Y : ndarray
        One dimension array of integer count-process observations
        Must have states ranging form 0 to N
    initial : ndarray
        Optional parameter initializing the guess for the hidden
        states. If none is provided, we will use a 2 distribution
        Poisson mixture model fit with EM. Please note that this
        procedure fails when the frequency of latent states is
        asymmetric, so you may want to provide different initial
        conditions.
    '''
    N = np.max(Y)+1  # Number of observation states
    O = np.arange(N) # List of available states

    if initial is None:
        classes = np.int32(Y>np.median(Y))
    else:
        classes = initial

    # Start with the density-based heuristic
    new_X = np.array(classes,np.int32)
    # Start with random state
    # new_X = np.array(urand(size=(len(counts),))<0.5,'float')
    X     = np.zeros(np.shape(new_X),'float')
    while not all(X==new_X):
        X[:] = new_X
        logP,logA,logB,params = hmm_poisson_parameter_guess(X,Y,N)
        new_X = hmm_viterbi(Y,logP,logA,logB)
        if any(map(hasNaN,(logP,logA,logB,X))):
            raise RuntimeError('NaN encountered')
    return X,params
