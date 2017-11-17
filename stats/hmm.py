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

from neurotools.stats.Gaussian import *

'''
Routines concerning discrete time hidden markov models
'''

import numpy as np
from neurotools.functions import slog
from neurotools.stats.distributions import poisson_logpdf
import random
from scipy.optimize import minimize
from math import factorial as fact
import scipy
from neurotools.stats.Gaussian import gaussian_quadrature
from neurotools.stats.Gaussian import gaussian_quadrature_logarithmic
from neurotools.functions import slog,sexp

def poisson_parameter_guess(X,Y,N):
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
        
    Returns
    -------
    logP : log Prior for state at first observation
    logA : log State transition array
    logB : log Poisson process rates
    params : p1,p01,p10,mu0,mu1
        p1  = np.mean(X)
        p01 = np.sum(np.diff(X)== 1)/(1+np.sum(X==0))
        p10 = np.sum(np.diff(X)==-1)/(1+np.sum(X==1))
        mu0 = np.mean(Y[X==0])
        mu1 = np.mean(Y[X==1])
    '''
    # Estimate model parameters from best-guess X
    p1  = np.mean(X)
    p01 = np.sum(np.diff(X)== 1)/(1+np.sum(X==0))
    p10 = np.sum(np.diff(X)==-1)/(1+np.sum(X==1))
    mu0 = np.mean(Y[X==0])
    mu1 = np.mean(Y[X==1])
    params = (p1,p01,p10,mu0,mu1)
    # Prior for state at first observation
    logP = np.array([slog(1-p1),slog(p1)])
    if not np.all(np.isfinite(logP)):
        raise RuntimeError(
            'Error computing marginal log-pr in Poisson data; zero rate?');
    # State transition array
    logA = np.array([
        [slog(1-p01), slog(p01)],
        [slog(p10), slog(1-p10)]],
        dtype=np.float64)
    if not np.all(np.isfinite(logA)):
        raise RuntimeError(
            'Error computing transition matrix');
    # Poisson process rates
    O = np.arange(N)      # List of available states
    logB = np.array([
        poisson_logpdf(O, mu0),
        poisson_logpdf(O, mu1)])
    if not np.all(np.isfinite(logB)):
        raise RuntimeError(
            'Error computing observation matrix');s
    return logP,logA,logB,params

def poisson_baum_welch(Y,initial=None):
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
    while not np.all(X==new_X):
        X[:] = new_X
        logP,logA,logB,params = poisson_parameter_guess(X,Y,N)
        new_X = viterbi_log(Y,logP,logA,logB)
        if any(map(hasNaN,(logP,logA,logB,X))):
            raise RuntimeError('NaN encountered')
    return X,params

def viterbi(Y,P,A,B):
    '''
    See https://en.wikipedia.org/wiki/Viterbi_algorithm

    Parameters
    ----------
    Y : 1D array
        Observations (integer states)
    P : array shape = (nStates ,)
        1D array of priors for initial state
    A : array (nStates,nStates)
        State transition matrix 
    B : ndarray K x N
        conditional probability matrix (emission/observation)
        probabilty of each observation given each state
    '''
    logP = np.log(P)
    logA = np.log(A)
    logB = np.log(B)
    return viterbi_log(Y,logP,logA,logB)

def viterbi_log(Y,logP,logA,logB):
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

def poisson_viterbi_state_infer(Y,initial=None):
    '''
    Fit Hidden Markov Model using np.expectation Maximization. For now
    it is limited to two latent states. The Viterbi algorithm is
    used to assign the most likely overall trajectory in the 
    np.expectation maximization. This is different from the Baum-Welch
    algorithm which uses the forward-backward algoirthm to infer
    distributions over states rather than the single most likely
    trajectory.

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
        
    Returns
    -------
    X : inferred states
    params : inferred model parameters
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
    while not np.all(X==new_X):
        X[:] = new_X
        logP,logA,logB,params = poisson_parameter_guess(X,Y,N)
        new_X = viterbi_log(Y,logP,logA,logB)
        # if X has degenerated to one class we have a problem
        # (we probably need soft-EM to compensate?)
        s = np.sum(new_X)
        if s==0 or s==len(new_X):
            raise RuntimeError('Inference collapsed to a single class; consider soft-EM instead?')
        if any(map(hasNaN,(logP,logA,logB,X))):
            raise RuntimeError('NaN encountered')
    return X,params


def forward_backward(y, x_0, T, B):
    '''
    forward_backward(y, x_0, T, B)
    
    Compute distribution of hidden states conditioned on all time
    points using the forward-backward algorithm.    

    Parameters
    ----------
    y : ndarray 
        An n_timepoints long vector of state observations
    x_0: ndarray
        An n_hidden_states long vector of initial state prior
    T : ndarray
        An n_hidden_states x n_hidden_states transition matrix
    B : 
        An n_hidden_states x n_observable_stated observation
        (emission) matrix

    Returns
    -------
    iterable f:
        forward inference results
    iterable b:
        backward inference results
    iterable pr:
        posterior inference results (forward * backward)

    Example
    -------
    ::
    
        # Initialize model
        n_states = 2
        x_0 = array([ 0.6,  0.4])
        T   = array([
               [ 0.69,  0.3 ],
               [ 0.4 ,  0.59]])
        B   = array([
               [ 0.5,  0.4,  0.1],
               [ 0.1,  0.3,  0.6]])
        # Initialize example states
        y  = array([0, 1, 2, 0, 0])
        fwd, bwd, posterior = forward_backward(y,x_0,T,B)
        print(posterior)
        # Verify that it works with a large number of observations
        y = randint(0,n_states,(10000,))
        fwd, bwd, posterior = forward_backward(y,x_0,T,B)
        print(posterior)
    '''   
    
    n_times   = len(y)
    n_states  = len(x_0)
    n_observe = B.shape[1]

    # Argument verification    
    assert(T.shape[0]==n_states)
    assert(B.shape[0]==n_states)
    assert(np.all(y>=0))
    assert(np.all(y<n_observe)) 
    
    # forward part of the algorithm
    # Compute conditional probability for each state
    # based on all previous observations
    f = np.zeros((n_times,n_states))
    # initial state conditioned on first observation
    f[0] = B[:,y[0]] * x_0 
    for i in range(1,n_times):
        # condition on transition from previous state
        # and current observation
        f[i] = B[:,y[i]]*np.ravel(np.dot(f[i-1],T))
        # normalize for numerical stability
        f[i]/= f[i].sum()
        assert np.all(np.isfinite(f[i]))
    # backward part of the algorithm
    # compute conditional probabilities of subsequent
    # chain from each state at each time-point
    b = np.zeros((n_times,n_states))
    # final state is fiexd with probability 1
    b[0] = 1;
    for i in range(1,n_times):             
        # combine next observation, and likelihood
        # of state based on all subsequent, weighted
        # according to transition matrix
        b[i] = np.dot(T,B[:,y[n_times-i]]*b[i-1])
        # normalize for numerical stability
        b[i]/= b[i].sum()
        assert np.all(np.isfinite(b[i]))
    # put the backward inferences in forward order
    b = b[::-1]
    # Merge the forward and backward inferences
    pr = f * b
    # Normalize the get a proper density
    pr /= pr.sum(1)[:,None]
    return f,b,pr

def jump(pobabilities):
    '''
    Markov jump function: pick a state according to probabilities

    Parameters
    ----------
    probabilities : vector of probabilities, must sum to 1.
    '''
    pobabilities = np.ravel(pobabilities)
    r = random.uniform(0,np.sum(pobabilities))
    cumulative = 0.
    for i,pr in enumerate(pobabilities):
        if cumulative+pr>=r: return i
        cumulative += pr
    assert False

def sample(L,T,B,x0):
    '''
    x,y = sample(L,T,B,x0)
    Sample from a discrete hidden markov model.
    Parameters
    ----------
    L : number of samples to draw
    T : state transition matrix
    B : observation (emission) matrix
    x0: initial conditions
    '''
    # Prepare to sample a path for the latent state
    x = np.zeros((L,),'int')
    x[0]=x0
    for i in range(1,L):
        x[i] = jump(T[x[i-1]])
    # Prepare observations from sample path
    y = np.zeros((L,),'int')
    for i in range(0,L):
        y[i] = jump(B[x[i]])
    return x,y

def log_likelihood(x,y,T,B,x0):
    '''
    Likelihood of hidden (x) and observed (y) state sequence for
    hidden markov model with hidden-state transition probabilities T
    and emission probabilities B, and initial state x0. 
    
    Returns the log likelihood, which is more numerically stable
    and avoids overflow
    '''
    L,n_hid = x.shape
    _,n_obs = B.shape
    
    # Validate arguments
    assert issubclass(x.dtype.type, np.floating)
    assert issubclass(y.dtype.type, np.integer)
    assert issubclass(T.dtype.type, np.floating)
    assert issubclass(B.dtype.type, np.floating)
    assert issubclass(x0.dtype.type, np.floating)
    assert y.shape[0]==L
    assert T.shape==(n_hid,n_hid)
    assert len(x0)==n_hid
    assert np.all(y>=0) and np.all(y<n_obs)
    assert np.all(B>0)
    assert np.all(T>0)
    assert np.all(x>0)
    assert np.all(x0>0)
    assert np.all(abs(x.sum(1)-1.)<1e-9)
    assert np.all(abs(B.sum(1)-1.)<1e-9)
    assert np.all(abs(T.sum(1)-1.)<1e-9)
    assert abs(x0.sum()-1.)<1e-9 
    
    log_likelihood = 0
    
    # likelihood of observations
    for i in range(L):
        log_likelihood += np.log(dot(x[i],B)[y[i]])
    
    # likelihood of transitions
    # for each current state
    #     take probability of current state
    #     for each following state
    #         weighted by density in this state
    #         take probability of transitioning to this state
    # In other words
    #
    # sum(T*x[i][:,None]*x[i+1][None,:])
    for i in range(L-1):
        log_likelihood += np.log(sum(T*x[i][:,None]*x[i+1][None,:]))
        
    return log_likelihood
    

def baum_welch(y,n_hid,convergence = 1e-10, eps = 1e-4, miniter=10):
    '''
    That,Bhat,X0hat,f,b,p,llikelihood = baum_welch(y,n_hid,convergence = 1e-6, eps = 1e-4)
    
    Baum-Welch algorithm
    
    Use np.expectation maximization to find locally optimal parameters for
    a hidden markov model.
    
    Parameters
    ----------
    
    y : 
        1D array of state observations
    n_hid : 
        number of hidden statees
    convergence : float, default 1e-6 
        stop when the largest change in tranisition 
        or emission matrix parameters is less than this value
    eps : float, default 1e-8 
        small uniform probability for regularization,
        to avoid probability of any state or transition going to zero
        
    Returns
    -------
    That : estimated transition matrix
    Bhat : estimated observation matrix
    X0hat : estimated initial state
    f : forward filter of observations with estimated model
    b : backward filter of observations with estimated model
    p : smoothing estimation of latent state using estimated model
    llikelihood : likelihood of data given model
    '''

    # Verify arguments
    ntime = len(y)
    n_obs = len(np.unique(y))
    assert n_hid>1
    assert n_obs>1
    print('%d hidden and %d observed states'%(n_hid,n_obs))
    if not (y.max()+1==n_obs):
        print('There are more state IDs than there are distinct states')
        print('Please reformat data so that states map to 0..N and every state is represented')
        assert y.max()+1==n_obs

    # Initialize random guess
    # Estimated transition operator between hidden states
    # Initialize to the identity
    That = np.eye(n_hid)*.9+np.ones((n_hid,n_hid))*.1/(n_hid-1)
    # Estimated transition operator between observed states
    # Initialize to uniform distribution
    Bhat = np.ones((n_hid,n_obs))/n_obs
    # Estimated initial conditions
    # Initialize to uniform distribution
    X0hat = np.ones(n_hid)/n_hid

    for ii in range(500):
        # Infer hidden states using current estimated parameters
        fwd,bwd,p = forward_backward(y,X0hat,That,Bhat)

        assert np.all(np.isfinite(fwd))
        assert np.all(np.isfinite(bwd))
        assert np.all(np.isfinite(p))

        # Construct new state estimates

        # Get the joint density of sucessive states x_n x_n+1
        # conditioned on the data
        E = np.zeros((n_hid,n_hid))
        for i in range(ntime-1):
            tr = fwd[i][:,None]*That*(bwd[i+1]*Bhat[:,y[i+1]])[None,:]
            E += (tr)/(sum(tr))
        # Divide this by the marginal density of states to get
        # the conditional density of x_n+1 based on x_n
        nThat  = E/(p.sum(0)[:,None]) + eps
        nThat /= nThat.sum(1)[:,None]

        # Initial condition is often easy to infer, and 
        # converges quickly.
        # But if we allow probability of any state to 
        # get too small, it causes numerical precision issues
        nX0hat = p[0]+eps
        nX0hat/= nX0hat.sum()
        nBhat  = np.zeros(Bhat.shape)
        for i in range(n_obs):
            nBhat[:,i] = (p[y==i].sum(0))/(p.sum(0))

        assert np.all(np.isfinite(nThat))
        assert np.all(np.isfinite(nX0hat))
        assert np.all(np.isfinite(nBhat))
        assert np.all(nBhat>0)
        assert np.all(nThat>0)
        assert np.all(nX0hat>0)

        delta = max(abs(That-nThat).max(),abs(nBhat-Bhat).max())
        That  = nThat
        Bhat  = nBhat
        X0hat = nX0hat
        llikelihood = log_likelihood(p,y,That,Bhat,X0hat)
        if (ii%1==0): 
            print(ii,'\n\tdelta=',delta,'\n\tllike=',llikelihood)
        if ii>miniter and delta<convergence: break
    #
    fwd,bwd,p = forward_backward(y,X0hat,That,Bhat)
    return That,Bhat,X0hat,fwd,bwd,p,llikelihood


def forward_abstract(y, x0, T, B):
    '''
    Abstracted form of forward filtering algorithm
    
    Parameters
    ----------
    y     : iterable of observations
        sequence of observations
    B     : y→(P(x)→P(x)), 
        observation model conditioning on observations P(x|y(t))
        should accept and observation, and return a function from 
        prior distribution to posterior distribution.
    x0    : P(x)
        initial state estimate (distribution)
    T.fwd : P(x)→P(x); 
        linear operator for the forward pass;
        should be a function that accepts and returns a distribution
    T.bwd : P(x)→P(x); 
        linear operator for the backward pass
        should be a function that accepts and returns a distribution
    '''
    L = len(y)
    # forward part of the algorithm
    # Compute conditional probability for each state
    # based on all previous observations
    f = {}
    # initial state conditioned on first observation
    f[0] = B(y[0]) * x0
    for i in range(1,L):
        # condition on transition from previous state and current
        # observation
        f[i] = B(y[i])*T.fwd(f[i-1])
    f = [f[i] for i in range(L)]
    return np.array(f)
    
def backward_abstract(y, x0, T, B):
    '''
    Abstracted form of backward filtering algorithm
    
    Parameters
    ----------
    y : iterable of observations
        sequence of observations
    B : y→(P(x)→P(x)), 
        observation model conditioning on observations P(x|y(t))
        should accept and observation, and return a function from 
        prior distribution to posterior distribution.
    x0: P(x)
        initial state estimate (distribution)
    T.fwd : P(x)→P(x); 
        linear operator for the forward pass;
        should be a function that accepts and returns a distribution
    T.bwd : P(x)→P(x); 
        linear operator for the backward pass
        should be a function that accepts and returns a distribution
    '''
    L = len(y)
    # backward part of the algorithm
    # compute conditional probabilities of subsequent
    # chain from each state at each time-point
    b = {}
    # final state is fixed with probability 1
    b[L-1] = f[L-1];
    for i in range(L-1)[::-1]:
        # combine next observation, and likelihood
        # of state based on all subsequent, weighted
        # according to transition matrix
        b[i] = T.bwd(B(y[i+1])*b[i+1])
    # Clean up the dictionary representations used for notational clarity above
    b = [b[i] for i in range(L)]
    return np.array(b)

def forward_backward_abstract(y, x0, T, B, prior=1):
    '''
    Abstracted form of forward-backward algorithm
    
    Parameters
    ----------
    y : iterable
        sequence of observations
    B : y→(P(x)→P(x)), 
        conditioning on observations P(x|y(t))
    x0: P(x)
        Initial condition
    T.fwd : P(x)→P(x) 
        Operator for the forward  pass
    T.bwd : P(x)→P(x)
        Operator for the backward pass
    prior : Optional, P(x) 
        Prior to be multiplied with the latent state on every time-step
    '''
    L = len(y)
    # forward part of the algorithm
    # Compute conditional probability for each state
    # based on all previous observations
    f = {}
    # initial state conditioned on first observation
    f[0] = B(y[0]) * x0
    for i in range(1,L):
        # condition on transition from previous state and current observation
        f[i] = B(y[i])*(T.fwd(f[i-1])*prior)
    # backward part of the algorithm
    # compute conditional probabilities of subsequent
    # chain from each state at each time-point
    b = {}
    # final state is fixed with probability 1
    b[L-1] = f[L-1];
    for i in range(L-1)[::-1]:
        # combine next observation, and likelihood
        # of state based on all subsequent, weighted
        # according to transition matrix
        b[i] = T.bwd(B(y[i+1])*(b[i+1]*prior))
    # Merge the forward and backward inferences
    pr = [f[i]*b[i] for i in range(L)]
    # Clean up the dictionary representations used for notational clarity above
    f = [f[i] for i in range(L)]
    b = [b[i] for i in range(L)]
    return np.array(f),np.array(b),np.array(pr)

class DiffusionGaussian:
    '''
    Diffusion operator to use in abstracted forward-backward
    Operates on Gaussian densities
    '''
    def __init__(s,d):
        '''
        d should be the variance of the process
        '''
        s.d = d
    def fwd(s,p):
        '''
        Forward (and backward) operator of a Gaussian diffusion process
        (Wiener process).
        
        Parameters
        ----------
        d : Gaussian object
        
        Returns
        -------
        result : new gaussian object reflecting diffusion
        '''
        result = Gaussian(p.m,p.t/(1.0+s.d*p.t))
        if hasattr(p,'lognorm'):
            result.lognorm = p.lognorm
        return result
    bwd = fwd
    __call__ = fwd
    __mul__ = fwd

class PoissonObservationApproximator(Gaussian):
    '''
    Approximate Gaussian distribution to use in abstracted forward-
    backward. Used to condition Gaussian states on Poisson 
    observations. Uses 1D integration (quadrature) to estimate 
    posterior moments. Assumes log λ = a*x+b.
    
    Parameters
    ----------
    a: log-rate gain parameter
    b: log-rate bias parameter
    y: The observed count (non-negative integer)
    '''
    def __init__(s,a,b,y):
        s.a,s.b,s.y = (a,b,y)
    def __mul__(s,o):
        # Estimate integration limits
        if o is 1 or o.t<1e-6:
            # No information about the state yet.
            # We aren't sure over what domain to perform the integration
            # Use a Laplace approximation of p(x|y) to get appx bounds
            m0 = (np.log(s.y+1)-s.b)/s.a
            t0 = (s.y+1)*s.a**2
            s0 = np.sqrt(1/t0)
        else:
            # Use the prior on x to guess integration limits
            m0 = o.m
            s0 = np.sqrt(1/o.t)
        # Integrate within ±4σ of the mean of the prior
        x = np.linspace(m0-4*s0,m0+4*s0,50)#
        
        # Poisson distribution
        # Best when rate is large and counts are frequenty >1
        # Get the conditional probability of state given this observation
        # this is poisson in lambda = np.exp(ax+b)
        # pxy = np.exp(s.y*(s.a*x+s.b)-np.exp(s.a*x+s.b))/scipy.special.gamma(s.y+1)
        # Bernoilli: stable when rate is very low
        # pxy = np.exp(s.y*(s.a*x+s.b))/(1+np.exp(s.a*x+s.b))
        # Multiply pxy by distribution o, 
        # handling identity as special case
        # p = pxy*(o if o is 1 else o(x)+1e-10)
        # p = np.float64(p) #Hmm
        # assert np.all(np.isfinite(p))
        # Estimate mean and variance of posterior
        # return gaussian_quadrature(p,x)
        
        #logpxy = s.y*(s.a*x+s.b)-np.exp(s.a*x+s.b) - scipy.special.loggamma(s.y+1)
        # ignore normalization
        logpxy = s.y*(s.a*x+s.b)-sexp(s.a*x+s.b)
        assert np.all(np.isfinite(logpxy))
        logpxy -= np.mean(logpxy)
        assert np.all(np.isfinite(logpxy))
        if not o is 1:
            logpxy += slog(o(x))
        assert np.all(np.isfinite(logpxy))
        logpxy -= np.mean(logpxy)
        assert np.all(np.isfinite(logpxy))
        p = sexp(logpxy)
        assert np.all(np.isfinite(p))
        # Estimate mean and variance of posterior
        return gaussian_quadrature(p,x)
        
    def __str__(s): 
        return 'Approximator(%s,%s,%s)'%(s.a,s.b,s.y)

class PoissonObservationModel:
    '''
    Poisson observation model
    Returns a density that, when multiplied by a Gaussian,
    returns a numeric approximation of the posterior as a Gaussian
    see: PoissonObservationApproximator
    '''
    def __init__(s,a,b):
        '''
        Parameters
        ----------
        a: log-rate gain parameter
        b: log-rate bias parameter
        '''
        s.a,s.b = a,b
    def __call__(s,y):
        return PoissonObservationApproximator(s.a,s.b,y)

class BernoulliObservationApproximator(Gaussian):
    '''
    Approximate Gaussian distribution to use in abstracted forward-
    backward. Used to condition Gaussian states on Poisson 
    observations
    '''
    def __init__(s,a,b,y):
        s.a,s.b,s.y = (a,b,y)
    def __mul__(s,o):
        # Estimate integration limits
        if o is 1 or o.t<1e-6:
            # No information about the state yet.
            # We aren't sure over what domain to perform the integration
            # Use a Laplace approximation of p(x|y) to get appx bounds
            m0 = (np.log(s.y+1)-s.b)/s.a
            t0 = (s.y+1)*s.a**2
            s0 = np.sqrt(1/t0)
        else:
            # Use the prior on x to guess integration limits
            m0 = o.m
            s0 = np.sqrt(1/o.t)
        # Integrate within ±4σ of the mean of the prior
        x = np.linspace(m0-4*s0,m0+4*s0,50)#
        # Get the conditional probability of state given this observation
        # Bernoilli may also be a little more stable sometime
        pxy = np.exp(s.y*(s.a*x+s.b))/(1+np.exp(s.a*x+s.b))
        # Multiply pxy by distribution o, 
        # handling identity as special case
        p = pxy*(o if o is 1 else o(x)+1e-10)
        assert np.all(np.isfinite(p))
        return gaussian_quadrature(p,x)
    def __str__(s): 
        return 'Approximator(%s,%s,%s)'%(s.a,s.b,s.y)
class BernoulliObservationModel:
    '''
    Bernoulli observation model
    Returns a density that, when multiplied by a Gaussian,
    returns a numeric approximation of the posterior as a Gaussian
    see: BernoulliObservationApproximator
    '''
    def __init__(s,a,b):
        s.a,s.b = a,b
    def __call__(s,y):
        return BernoulliObservationApproximator(s.a,s.b,y)

class TruncatedPoissonObservationApproximator(Gaussian):
    '''
    Approximate Gaussian distribution to use in abstracted forward-
    backward. Used to condition Gaussian states on Poisson 
    observations
    '''
    def __init__(s,a,b,y):
        s.a,s.b,s.y = (a,b,y)
    def __mul__(s,o):
        # Estimate integration limits
        if o is 1 or o.t<1e-6:
            # No information about the state yet.
            # We aren't sure over what domain to perform the integration
            # Use a Laplace approximation of p(x|y) to get appx bounds
            m0 = (np.log(s.y+1)-s.b)/s.a
            t0 = (s.y+1)*s.a**2
            s0 = np.sqrt(1/t0)
        else:
            # Use the prior on x to guess integration limits
            m0 = o.m
            s0 = np.sqrt(1/o.t)
        # Integrate within ±4σ of the mean of the prior
        x = np.linspace(m0-4*s0,m0+4*s0,50)#
        # Get the conditional probability of state given this observation
        # this is poisson in lambda = np.exp(ax+b)
        #pxy = np.exp(s.y*(s.a*x+s.b)-np.exp(s.a*x+s.b))/scipy.special.gamma(s.y+1)
        # Bernoilli may also be a little more stable sometime
        #pxy = np.exp(s.y*(s.a*x+s.b))/(1+np.exp(s.a*x+s.b))
        # Truncated Poisson
        ll  = s.a*x+s.b 
        l   = np.exp(ll)
        pxy = np.exp(s.y*ll)/(1+l+l**2*.5)
        # Multiply pxy by distribution o, 
        # handling identity as special case
        p = pxy*(o if o is 1 else o(x)+1e-10)
        assert np.all(np.isfinite(p))
        return gaussian_quadrature(p,x)
    def __str__(s): 
        return 'Approximator(%s,%s,%s)'%(s.a,s.b,s.y)
class TruncatedPoissonObservationModel:
    '''
    Poisson observation model
    Returns a density that, when multiplied by a Gaussian,
    returns a numeric approximation of the posterior as a Gaussian
    see: PoissonObservationApproximator
    '''
    def __init__(s,a,b):
        s.a,s.b = a,b
    def __call__(s,y):
        return TruncatedPoissonObservationApproximator(s.a,s.b,y)

from numpy.linalg import solve

class MVGaussian:
    '''
    Multivariate Gaussian model to use in abstracted forward-backward
    '''
    def __init__(s,M,T,TM=None):
        '''
        M vector of means
        T precision matrix
        TM (optional) precomputed product of precision and mean
        '''
        s.M,s.T = M,T
        s.TM = np.dot(T,M) if TM is None else TM
    def __mul__(s,o):
        if o is 1: return s
        assert isinstance(o,MVGaussian)
        assert np.all(np.isfinite(o.T))
        assert np.all(np.isfinite(o.M))
        # Precision matricies add
        T = s.T+o.T
        # Linearly combine means weighted by precision
        TM = s.TM + o.TM
        # Recover mean vector via linear system solver
        # M=np.dot(np.linalg.inv(T),TM) ==> T*M=TM
        if abs(np.linalg.det(T))<1e-20:
            # Singular?
            print('MVGaussian: singular precision matrix!')
            M = TM
        else:
            M = np.linalg.solve(T,TM)
        assert np.all(np.isfinite(T))
        assert np.all(np.isfinite(M))
        return MVGaussian(M,T,TM)
    __call__ = lambda s,x: np.exp(-0.5*s.t*(x-s.m)**2)*np.sqrt(s.t/(2*np.pi))

class MVGUpdate():
    '''
    A: linear system transition matrix. Means evolve as X = AX
    sigma: linear system covariance transition. it is a matrix.
    '''
    def __init__(s,A,sigma):
        s.A = A
        s.B = np.linalg.inv(A)
        s.sigma = sigma
    def fwd(s,o):
        assert isinstance(o,MVGaussian)
        assert np.all(np.isfinite(o.T))
        assert np.all(np.isfinite(o.M))
        M = s.A.dot(o.M)
        T = o.T
        T = s.B.T.dot(T.dot(s.B))
        D = T.shape[0]
        R = np.eye(D) + T.dot(s.sigma)
        T = np.linalg.solve(R,T)
        assert np.all(np.isfinite(T))
        assert np.all(np.isfinite(M))
        return MVGaussian(M,T)
    def bwd(s,o):
        assert isinstance(o,MVGaussian)
        assert np.all(np.isfinite(o.T))
        assert np.all(np.isfinite(o.M))
        M = o.M.dot(s.A)
        T = o.T
        D = T.shape[0]
        R = np.eye(D) + T.dot(s.sigma)
        T = np.linalg.solve(R,T)
        T = s.A.T.dot(T.dot(s.A))
        assert np.all(np.isfinite(T))
        assert np.all(np.isfinite(M))
        return MVGaussian(M,T)

def lgcp_observation_minimizer(y,px,B):
    def fun(x):
        lograte = B.dot(x)
        rate = np.exp(lograte)
        x_mu = x-px.M
        lgpr = y*lograte - rate - 0.5*x_mu.dot(px.T.dot(x_mu))
        return -lgpr
    def jac(x):
        return (x-px.M).dot(px.T)-B*(y-np.exp(B.dot(x)))
    def hess(x):
        return np.outer(B,B.T)*np.exp(B.dot(x)) + px.T
    return fun,jac,hess
    
class MVPoissonObservation():
    '''
    y: binary observations
    B: projection vector from multivariate system to log-rate
    
    Uses Laplace approximation for covariance
    '''
    class MVPoissonApproximator():
        def __init__(s,B,y):
            s.B,s.y=B,y
        def __mul__(s,o):
            assert isinstance(o,MVGaussian)
            assert np.all(np.isfinite(o.T))
            assert np.all(np.isfinite(o.M))
            # Get mode and covariance
            fun,jac,hess = lgcp_observation_minimizer(s.y,o,s.B)
            mode  = minimize(fun, o.M, jac=jac, hess=hess, method='Newton-CG')
            return MVGaussian(mode.x,hess(mode.x))
    def __init__(s,B):
        s.B = B
    def __call__(s,y):
        return s.MVPoissonApproximator(s.B,y)

class OUGaussian:
    def __init__(s,var,tau,dt,regularize):
        s.params = var,tau,dt
        s.regularize = regularize
    def fwd(s,p):
        var,tau,dt = s.params
        a = np.exp(-dt/tau)
        result = Gaussian(a*p.m,1./(a*a/p.t+var+s.regularize))
        return result
    def bwd(s,p):
        var,tau,dt = s.params
        a = np.exp(-dt/tau)
        # Needed for regularization, a must stay close to 1 for stability
        a += s.regularize
        result = Gaussian(p.m/a,1./(1./p.t/(a*a)+var))
        return result

class MVGOUUpdate():
    '''
    A: linear system transition matrix. Means evolve as X = AX
    sigma: linear system covariance transition. it is a matrix.
    '''
    def __init__(s,A,mean,sigma,regularize):
        s.A = A 
        A = T.shape[0]
        s.B = np.linalg.inv(A + np.eye(D)*regularize)
        s.sigma = sigma
        s.regularize = regularize
        s.mean = mean
    def fwd(s,o):
        assert isinstance(o,MVGaussian)
        assert np.all(np.isfinite(o.T))
        assert np.all(np.isfinite(o.M))
        M = s.A.dot(o.M-s.mean)+s.mean
        T = o.T
        T = s.B.T.dot(T.dot(s.B))
        D = T.shape[0]
        R = np.eye(D) + T.dot(s.sigma)
        T = np.linalg.solve(R,T)
        assert np.all(np.isfinite(T))
        assert np.all(np.isfinite(M))
        return MVGaussian(M,T)
    def bwd(s,o):
        assert isinstance(o,MVGaussian)
        assert np.all(np.isfinite(o.T))
        assert np.all(np.isfinite(o.M))
        M = (o.M-s.mean).dot(s.A)+s.mean
        T = o.T
        D = T.shape[0]
        R = np.eye(D) + T.dot(s.sigma)
        T = np.linalg.solve(R,T)
        T = s.A.T.dot(T.dot(s.A))
        assert np.all(np.isfinite(T))
        assert np.all(np.isfinite(M))
        return MVGaussian(M,T)
        
        
