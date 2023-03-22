#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Routines concerning information theory
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import neurotools.signal
from collections import defaultdict
from neurotools.stats.distributions import poisson_pdf
from neurotools.signal import unitsum

from neurotools.util.array import verify_axes, axes_complement, reslice
from scipy.linalg import circulant

from neurotools.util.tools import piper
@piper
def bits_to_nats(b):
    return np.log(2)*b #2**b = e**n
@piper
def nats_to_bits(n):
    return np.log2(np.e)*n #2**b = e**n
b2n = bits_to_nats
n2b = nats_to_bits

def _masked_log(x):
    return np.log(x, out=np.zeros_like(x), where=(x!=0))

class discrete:
    '''
    Collected methods for calculating biased information-theoretic
    quantities directly from discrete (categorical) probability
    distributions. 
    These have never been fully tested.
    '''

    @classmethod
    def DKL(cls,P,Q,axis=None):
        '''
        Compute KL divergence D(P||Q) 
        between discrete distributions ``P`` and ``Q``.
        
        Parameters
        ----------
        P : np.array
            Vector of probabilities
        Q : np.array
            Vector of probabilities
        
        Returns
        -------
        DKL : float
            KL divergence from P to Q
        '''
        if np.shape(P)!=np.shape(Q): raise ValueError(
            'Arrays P and Q must be the same shape')
        Q  = np.float64(Q)
        if np.any(Q<=0.0): raise ValueError(
            'Q cannot contain zeros')
        P = unitsum(P,axis=axis)
        Q = unitsum(Q,axis=axis)
        return np.sum(P*(_masked_log(P)-np.log(Q)),axis=axis)
        
    @classmethod
    def H(cls,p,axis=None):
        '''
        Sample entropy ``-Σ p ln(p)`` in nats.
        
        Parameters
        ----------
        p: array-like numeric
            List of frequencies or counts
        
        Returns
        -------
        :float
            Shannon entropy of discrete distribution 
            with observed `counts`
        '''
        p = unitsum(p,axis=axis)
        return -np.sum(p*_masked_log(p),axis=axis)
        '''
        Test code
        print(n2b<<discrete.H([1,1]),'?=',1)
        print(n2b<<discrete.H([1,1,1,1]),'?=',2)
        print(n2b<<discrete.H([2,1,1]),'?=',3/2)
        1.0 ?= 1
        2.0 ?= 2
        1.4999999999999998 ?= 1.5
        '''
    
    @classmethod
    def H_samples(cls,samples):
        '''
        Calculate sample entropy ``-Σ p ln(p)`` (in nats)
        from a list of samples.
        
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
        return cls.H(counts.values())
        
    @classmethod
    def Hcond(cls,p,axis=None):
        '''
        Conditional entropy :math:`H_{y|x}`. 
        
        Parameters
        ----------
        p: array-like numeric
            List of frequencies or counts
        axis: tuple
            List of axes corresponding to ``y`` in 
            :math:`H_{y|x}`. 
            Remaining axes presumed to correspond to ``x``.
        
        Returns
        -------
        :float
            Shannon entropy of discrete distribution 
            with observed `counts`
        '''
        p = unitsum(p)
        if len(p.shape)<=1:
            raise ValueError('p should be at least 2D')
        if axis is None:
            if len(p.shape)>2:
                raise ValueError(
                    'p has more than two dimensions; '
                    'please specify the list of axes corresponding '
                    'to one of the variables of interest to calculate '
                    'joint conditional entropy')
            # Default to the first axis otherwise
            axis = (0,)
        
        Px   = np.sum(p,axis=axis)
        Pygx = unitsum(p,axis=axis)
        return np.sum( Px * (-np.sum(Pygx*_masked_log(Pygx),axis=axis)))
        '''
        Test code
        # row, column
        # H_{y|x}
        # columns are conditioned variable
        # rows are conditional probabilities
        p_test = [
            [1,1,1,1],
            [1,1,1,1],
            [2,2,2,2]
        ]
        print(n2b<<discrete.Hcond(p_test,0),'?=',3/2)
        print(n2b<<discrete.Hcond(p_test,1),'?=',2)
        1.4999999999999998 ?= 1.5
        2.0 ?= 2
        '''

    @classmethod
    def I(cls,p,axes1=None,axes2=None):
        '''
        Mutual information from a discrete PDF
        
        Parameters
        ----------
        p: np.ndarray
            Array of probabilities or counts, at lest two dimensional.
        axes1: tuple
            List of axes corresponding to the first set of variables.       
        axes2: tuple
            List of axes corresponding to the second set of variables.       
        '''
        p = unitsum(p)
        if len(p.shape)<=1:
            raise ValueError('p should be at least 2D')
        
        # Validate first set of axes
        if axes1 is None:
            if len(p.shape)>2:
                raise ValueError(
                    'p has more than two dimensions; '
                    'please specify the list of axes corresponding '
                    'to one of the variables of interest to calculate '
                    'joint mutual information')
            if not axes2 is None:
                raise ValueError(
                    'If you want to use `axes2` to specify the '
                    'axes for the second variable, please also pass '
                    '`axes` to indicate the axes for the first')
            # Default to the first axis otherwise
            axes1 = (0,)
            
        # Validate second set of axes
        if axes2 is None:
            axes2 = axes_complement(p,axes1)
        elif len({*axes1}&{*axes2}):
            raise ValueError((
                'Lists of axes (%s,%s) for first and second '
                'variables overlap.')%(axes2,axes2))

        axes12 = tuple(axes1)+tuple(axes2)
        p = unitsum(p,axes12)

        # Marginal entropies
        # TODO: better way?
        ax2less1 = [i for i,a in enumerate(axes_complement(p,axes1)) if a in axes2]
        ax1less2 = [i for i,a in enumerate(axes_complement(p,axes2)) if a in axes1]
        Hy = cls.H(np.sum(p,axes1),tuple(ax2less1))
        Hx = cls.H(np.sum(p,axes2),tuple(ax1less2))
        
        # Joint entropy
        Hxy = cls.H(p,axes12)
        return Hx + Hy - Hxy
        
        '''
        p_test = [
            [1,1,1,1],
            [1,1,1,1],
            [2,2,2,2]
        ]
        print(n2b<<I(p_test),'?=',0)
        p_test = [
            [1,1,2,4],
            [1,2,4,1],
            [2,4,1,1],
            [4,1,1,2],
        ]
        print(n2b<<I(p_test),'?=',.25)
        0.0 ?= 0
        0.25000000000000006 ?= 0.25
        '''

    @classmethod
    def shuffle(cls,p,axes=(0,1)):
        '''
        Replace the joint density for the variables
        in ``axes`` with the product of their marginals, 
        all conditioned on any remaining variables not
        included in ``axes``.
        '''
        # Ensure normalized
        p = unitsum(p,axes)
        naxes = len(p.shape)
        q = np.ones(p.shape)
        for i in axes:
            others = tuple({*axes}-{i})
            q *= np.sum(p,axis=others)[reslice(naxes,others)]
        return q
        '''
        p = [[[1,2],
              [2,4]],
             [[1,4],
              [2,2]
             ]]
        p = array(p).transpose(1,2,0)
        discrete.shuffle(p)
        '''
    
    @classmethod
    def Ishuff(cls,p,axes=(0,1)):
        '''
        The shuffled information between
        (neuron₁, neuron₂, stimulus),
        in nats.
        
        Parameters
        ----------
        p: np.ndarray
            a 3D array of joint probabilities.
            The first two axes should index neuronal responses. 
            The third axis should index the extrinsic covariate.
        '''
        return cls.I(cls.shuffle(p,axes=axes),axes)
        '''
        p = [
            [[1,2],
             [2,4]],
            [[1,2],
             [2,4]]
            ]
        p = array(p).transpose(1,2,0)
        print(Ishuff(p))
        p = [
            [[1,2],
             [2,4]],
            [[2,4],
             [2,1]]
            ]
        p = array(p).transpose(1,2,0)
        print(n2b<<Ishuff(p))
        print(n2b<<cls.I(p,(0,1)))
        0.0
        0.09005577569707414
        0.11809427883737673
        '''

    @classmethod
    def deltaIshuff(cls,p,axes=(0,1)):
        '''
        Mutual information between 
        (neuron₁, neuron₂) and (stimulus),
        relative to a conditionally-shuffled
        baseline where
        P(neuron₁, neuron₂)
        is replaced by
        P(neuron₁) P(neuron₂),
        in nats.
        
        Naïve calculation of 
        Latham and Nirenberg equation (2).
        '''
        return cls.I(p,axes) - cls.Ishuff(p,axes)
    
    @classmethod
    def deltaI(cls,p,axes=(0,1)):
        '''
        Mutual information between 
        (neuron₁, neuron₂) and (stimulus),
        Minus the mutual information
        I(neuron₁, stimulus) and
        I(neuron₂, stimulus).
        
        Naïve calculation of 
        Latham and Nirenberg equation (5)
        from 
        Brenner et al., 2000; 
        Machens et al., 2001; 
        Schneidman et al., 2003.
        '''
        p = unitsum(p)
        
        # Separate axes into stimulus and neuron dimensions
        neur = axes
        stim = axes_complement(p,axes)
        
        # Marginals for stimulus s and neurons r
        Ps = np.sum(p,neur)
        Pr = np.sum(p,stim)
        
        # Joint stim|neurons
        Psgr = unitsum(p,stim)

        # Conditionally shuffled neuronal responses
        q = cls.shuffle(p,axes)
        
        # Shuffled (joint) stim|neurons
        Psgrshuff = unitsum(q,stim)
        
        return cls.DKL(p,q)
    
    @classmethod
    def deltaInoise(cls,p,axes=(0,1)):
        '''
        :math:`I_{r_1,r_2;s} - I^{\\text{shuffle}}_{r_1,r2;s}`.
        
        Naïve calculation of 
        Schneidman, Bialek, Berry (2003) equation (14).
        '''
        Isr = cls.I(p,axes)
        Jsr = cls.Ishuff(p,axes)
        return Isr - Jsr
    
    @classmethod
    def deltaIsig(cls,p,axes=(0,1)):
        '''
        :math:`I_{r_1,r_2;s} - I^{\\text{shuffle}}_{r_1,r2;s}`.
        
        Naïve calculation of 
        Schneidman, Bialek, Berry (2003) equation (15).
        '''
        result = -cls.Ishuff(p,axes)
        for others in circulant(axes)[:,1:]:
            result += cls.I(np.sum(p,axis=tuple(others)))
        return result
    
    @classmethod
    def syn(cls,p,axes=(0,1)):
        '''
        Mutual information between 
        (neuron₁, neuron₂) and (stimulus),
        Minus the mutual information
        I(neuron₁, stimulus) and
        I(neuron₂, stimulus).
        
        Naïve calculation of 
        Latham and Nirenberg equation (4)
        from 
        Brenner et al., 2000; 
        Machens et al., 2001; 
        Schneidman et al., 2003.
        '''
        result = cls.I(p,axes)
        for others in circulant(axes)[:,1:]:
            result -= cls.I(np.sum(p,tuple(others)))
        return result
    
    @classmethod
    def syn2(cls,p):
        '''
        Naïve calculation of 
        Schneidman, Bialek, Berry (2003) equation (11).
        
        This should match ``discrete.syn``
        
        This is fixed: p should be 3D with axes (0,1) the neurons
        and axis 2 the stimulus
        '''
        p = unitsum(p)
        
        # Mutual information for bivariate neuron marginal
        Ineur_marginal = cls.I(np.sum(p,2))
        
        # Average mutual information over stimuli
        ps = np.sum(p,(0,1))   
        Ineur_stim = np.sum(ps*cls.I(p,(0,),(1,)))
        
        return Ineur_stim - Ineur_marginal
        

def poisson_entropy_nats(l):
    '''
    Approximate entropy of a Poisson distribution in nats
    
    Parameters
    ----------
    l: positive float
        Poisson rate parameter
    '''
    cutoff = int(np.ceil(l+4*np.sqrt(l)))+1
    p = poisson_pdf(arange(cutoff),l)
    return discrete_entropy_distribution(p)


def betapr(k,N):
    '''
    Baysian estimation of rate `p` in Bernoulli trials.
    This returns the posterior median for `p` given 
    `k` positive examples from `N` trials, using Jeffery's
    prior. 
    
    Parameters
    ----------
    k: in or np.int23
        Number of observations for each state
    N: positive int
        ``N>k`` total observations
        
    Returns
    -------
    p: float or np.float32
        Estiamted probability or probability per bin, if
        ``k`` is a ``np.int32`` array. Probabilities are
        normalized to sum to 1.
    '''
    a = .5 + k
    b = N - k + .5
    p = a/(a+b)
    if np.size(p)>1:
        p*=(1./np.sum(p))
    return p

def beta_regularized_histogram_mutual_information(
    x,
    y,
    nb  = 4,
    eps = 1e-9,
    plo = 2.5,
    phi = 97.5,
    nshuffle=1000,
    nbootstrap=1000,
    ):
    '''
    A quick and dirt mutual information estimator. 
    
    The result will depend on the bin size but a quick
    suffle control provides a useful chance level. 
    
    Parameters
    ----------
    x: iterable<number>
        Samples for first variable ``x``
    y: iterable<number>
        Samples for second variable ``y``
    
    Other Parameters
    ----------------
    nb: positive int; default 4
        Number of bins. I suggest 3-10.
    eps: small positive float; default 1e-9
    plo: number ∈(0,100); default 2.5
    phi: number ∈(0,100); default 97.5
    
    Returns
    -------
    Ixy: float
        Histogram-based MI estimate 
        `Ixy = Hx + Hy - Hxy`
    Idelta: float
        Shuffle-adjust MI 
        `Idelta = np.median(Hx+Hy) - Hxy`
    pvalue: float
        p-value for significant MI from the shuffle test
    lo: float
        The boostrap `plo` percentil of `Idelta`
    hi: float
        The boostrap `phi` percentil of `Idelta`
    
    '''
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    assert len(x)==len(y)
    #x = nanrankdata(x)
    #y = nanrankdata(y)
    x = neurotools.signal.uniformize(x)
    y = neurotools.signal.uniformize(y)
    
    bins = [-eps]+[*np.arange(nb)[1:]/nb]+[1+eps]
    i = np.digitize(x,bins)-1
    j = np.digitize(y,bins)-1

    n0 = i*nb
    n  = n0+j
    N  = len(n)
    n2 = np.concatenate([n,np.arange(nb**2)])
    k  = np.unique(n2,return_counts=True,)[1]-1
    Hxy= betaH(k,N)
    H0 = 2*np.log(nb) # Hx = Hy = -nb*(1/nb)*np.log(1/nb)
    Ixy = H0 - Hxy

    # Shuffle control
    e = []
    for shuff in range(nshuffle):
        n2[:N] = n0 + np.random.choice(j,N,replace=False)
        k = np.unique(n2,return_counts=True,)[1]-1
        e.append(betaH(k,N))
    e = np.float32(e)
    pvalue = betapr(np.sum(e<Hxy),nshuffle)
    Idelta = np.median(e) - Hxy
    
    # Bootstrap uncertainty
    b = []
    for shuff in range(nbootstrap):
        n2[:N] = np.random.choice(n,N,replace=True)
        k = np.unique(n2,return_counts=True,)[1]-1
        p = betapr(k,N)
        b.append(betaH(k,N))
    Ib = H0 - np.float32(b)
    lo,hi = np.nanpercentile(Ib,[plo,phi])-(Ixy-Idelta)
    
    return Ixy, Idelta, pvalue, lo, hi
    
    
    
    
    
    
    
    




from typing import NamedTuple
class JointCategoricalDistribution(NamedTuple):
    counts: np.ndarray
    joined: np.ndarray
    kept  : np.ndarray
    states: np.ndarray
    nstate: np.ndarray

def joint(*args,
    nstates=None,
    remove_empty=False
    ):
    '''
    Convert a list of samples from several categorical
    variables in a single, new categorical variable. 

    This drops marginal states not present in any variable
    (which may not be what you want)
    '''
    # Collect arguments (each argument one variable)
    args = [np.array([*a]).ravel() for a in args]
    
    if remove_empty:
        # Convert discrete states to numeric labels for each variable
        # `states` contains the list of unique values
        # `index`  says the value # of each sample 
        states, index = [*zip(*[np.unique(a,return_inverse=1) for a in args])]
    else:
        # I will trust that you've given me integer counts
        if nstates is None: 
            nstates = [1+np.max(a) for a in args]
        states = [np.arange(n) for n in nstates]
        index  = args
        
    if nstates is None:
        # Count the number of states in each variable
        nstates = [*map(len,states)]
    
    # Encode each joint state as an integer
    # i.e. flatten joint space
    # (Equivalently: could have run `unique` on the
    # vectors of states)
    joined = index[0]
    for n,i in zip(nstates[1:],index[1:]):
        joined = joined*n + i
    
    if remove_empty:
        # Count each state
        kept,counts = np.unique(joined,return_counts=1)
    else:
        nkept  = np.prod(nstates)
        kept   = np.arange(nkept)
        counts = np.bincount(joined,minlength=nkept)
   
    return JointCategoricalDistribution(counts, joined, kept, states, nstates)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class dirichlet_density:
    '''
    Model empirical (sampled) densities of categorical
    variables using a Dirichlet prior. 
    
    The default prior is α = 1/k, where
    k is the number of states. This makes things behave
    nicer under marginalization. Specify bias=0.5 if 
    you want Jeffreys prior. 
    
    These are biased. 
    '''
    
    @classmethod
    def joint_model(cls, *args,bias=None):
        '''
        Build dirichlet model of joint categorical distribution
        '''
        from scipy.stats import dirichlet
        counts = joint(*args).counts
        if bias is None: bias = 1.0/len(counts)
        return dirichlet(counts+bias).mean()
    
    @classmethod
    def p(cls, *samples, bias=.5):
        '''
        Expected probabilty
        '''
        counts = joint(*samples).counts
        if bias is None: bias = 1.0/len(counts)
        α  = counts + bias
        α0 = np.sum(α)
        return α/α0
    
    @classmethod
    def lnp(cls, *samples, bias=.5):
        '''
        Expected log-probability 
        '''
        from scipy.special import digamma as ψ
        counts = joint(*samples).counts
        if bias is None: bias = 1.0/len(counts)
        α  = counts + bias
        α0 = np.sum(α)
        return ψ(α) - ψ(α0)
    
    @classmethod
    def plnp(cls,*samples,bias=None):
        '''
        Expected p*ln(p) 
        '''
        from scipy.special import digamma as ψ
        counts = joint(*samples).counts
        if bias is None: bias = 1.0/len(counts)
        α  = counts + bias
        α0 = np.sum(α)
        return α/α0*(ψ(α+1)-ψ(α0+1))
    
    @classmethod
    def H(cls,*samples,bias=None):
        '''
        Expected <-p*ln(p)> 
        '''
        return -np.sum(cls.plnp(*samples,bias=bias))

    @classmethod
    def I(cls,a,b,bias=None):
        '''
        Mutual information
        '''
        H = cls.H
        return H(a,bias=bias) + H(b,bias=bias) - H(a,b,bias=bias)
        '''
        # debug/explore code
        i1 = randint(0,5,1000)
        i2 = randint(0,5,1000)
        print(dd.H(i1,i2), dd.H(i1)+dd.H(i2))
        print(dd.H(i1,i2,bias=.5), dd.H(i1,bias=.5)+dd.H(i1,bias=.5))
        print(dd.H(i1,i2,bias=0), dd.H(i1,bias=0)+dd.H(i1,bias=0))
        '''

    @classmethod
    def redundancy(cls,x1,x2,y,bias=None):
        '''
        For ``(x1,x2,y)`` calculate
        ``I(x1,y) + I(x2,y) - I(joint(x1,x2),y)``.
        
        positive: redundant
        zero: independent
        negative: synergistic
        '''
        x12 = joint(x1,x2).joined
        I = cls.I
        return I(x1,y,bias=bias) + I(x2,y,bias=bias) - I(x12,y,bias=bias)
    
    @classmethod
    def foo(cls,x1,x2,y):
        counts, states, nstate = joint(x1,x2,y)
    
        

        
        
