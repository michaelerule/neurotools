#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
# more py2/3 compat
from neurotools.system import *

import neurotools.models.rbm.rbm as rb
from collections import defaultdict
from scipy.stats import entropy
from neurotools.jobs.ndecorator import memoize
import numpy as np

from neurotools.functions import slog, sexp, g, f, f1, f2

def conditional_map(k,v):
    '''
    Given two one-dimensional numpy arrays of the same length,
    containsing *keys* and *values*, return a dictionary mapping
    each key to the list of associated values. 
    
    newer versions of np.unique offer this functionality and are
    recommended if available.
    '''
    k = np.array(k)
    v = np.array(v)
    assert v.shape==k.shape
    assert len(v.shape)==1
    order  = np.argsort(k)
    k      = k[order]
    v      = v[order]
    idx    = np.where(np.diff(k)!=0.0)[0]+1
    starts = np.concatenate([[0],idx])
    stops  = np.concatenate([idx,[len(k)]])
    return {k[a]:v[a:b] for (a,b) in zip(starts,stops)}

class RBM_sample:

    #######################################################################
    # Initializers

    def __init__(s,S,W,Bh,Bv):
        '''
        Initialize RBM experiment sample output object
        
        Parameters
        ----------
        samples: array-like nsamples x (nvisible+nhidden)
        W: array-like Nh x Nv of RBM model weights
        Bh : length Nv vector of hidden-layer biases
        BV : length Nh vector of visible-layer biases
        '''
        s.Nh = Bh.shape[0]
        s.Nv = Bv.shape[0]
        s.Ns = S.shape[0]
        if not S.shape[1]==s.Nh+s.Nv:
            raise ValueError('Samples should be Nsample x Nvisible+Nhidden')
        if not W.shape==(s.Nh,s.Nv):
            raise ValueError('Weight matrix should be Nhidden x Nvisible in shape')
        if s.Nh>126 or s.Nv>126:
            raise ValueError('No more than 126 hidden or visible units supported')
        s.Sv,s.Sh = S[:,:s.Nv],S[:,s.Nv:]
        # Store object state
        s.S  = S
        s.W  = W
        s.Bh = Bh
        s.Bv = Bv
        s.precompute()
    
    def precompute(s):
        '''
        Precompute useful values. Unique patterns in samples, visibles,
        and their hash values. Marginal probabilities from samepls.
        '''        
        # Estimate marginal probabilities and corresponding activations
        s.Ph = np.mean(s.Sh,0)
        s.Pv = np.mean(s.Sv,0)
        s.Ah = rb.p2a(s.Ph)
        s.Av = rb.p2a(s.Pv)
        # Entropy capacity assuming factorial distribution
        s.Hhmax = np.sum(rb.bitent(s.Ph))
        s.Hvmax = np.sum(rb.bitent(s.Pv))
        # Unique identifiers based on bits
        s.Svi = rb.hashbits(s.Sv,s.Nv)
        s.Shi = rb.hashbits(s.Sh,s.Nh)
        # Empirical distribution of marginals from samples
        s.Uhi,s.Suidxh,s.Kh = np.unique(s.Shi,return_inverse=True,return_counts=True)
        s.Uvi,s.Suidxv,s.Kv = np.unique(s.Svi,return_inverse=True,return_counts=True)
        # Unique states observed in samples
        s.Uh = rb.unhashbits(s.Uhi,s.Nh)
        s.Uv = rb.unhashbits(s.Uvi,s.Nv)
        # Empirical probabilities -- using dithered prior
        s.Qh  = (s.Kh+np.random.rand(*s.Kh.shape)-0.5)
        s.Qh /= np.sum(s.Qh)
        s.Qv  = (s.Kv+np.random.rand(*s.Kv.shape)-0.5)
        s.Qv /= np.sum(s.Qv)
        # Empirical energies of unique states in sample
        s.Evs = -slog(s.Qv)
        s.Ehs = -slog(s.Qh)
        # Lowest observed energy in sample
        s.E0vs = np.min(s.Ev_unnormalized(s.Uv))
        s.E0hs = np.min(s.Eh_unnormalized(s.Uh))
        
    #######################################################################
    # Methods
        
    def Ahv(s,v):
        '''
        Activation of hiddens conditioned on visible pattern
        '''
        return rb.Acond(v,s.W,s.Bh)
        
    def Phv(s,v):
        '''
        Probability of hiddens conditioned on visible pattern
        '''
        return rb.Pcond(v,s.W,s.Bh)
        
    def Avh(s,h):
        '''
        Activation of visibles conditioned on hidden pattern
        '''
        return rb.Acond(h,s.W.T,s.Bv)
        
    def Pvh(s,h): 
        '''
        Probability of visibles conditioned on hidden pattern
        '''
        return rb.Pcond(h,s.W.T,s.Bv)
        
    def Eh_unnormalized(s,h):
        '''
        Energy of hidden state, up to a constant. This constant depends 
        on the joint model partition function (normalization constant) and
        is difficult to compute in general.
        '''
        if h.shape[0]!=s.Nh:
            h = h.T
        marginal    = -s.Bh.dot(h)
        conditional = -np.sum(g(s.W.T.dot(h).T+s.Bv).T,axis=0)
        return marginal + conditional
        
    def Ev_unnormalized(s,v):
        '''
        Energy of visible state, up to a constant. This constant depends 
        on the joint model partition function (normalization constant) and
        is difficult to compute in general.
        '''
        if v.shape[0]!=s.Nv:
            v = v.T
        marginal    = -s.Bv.dot(v)
        conditional = -np.sum(g(s.W.dot(v).T+s.Bh).T,axis=0)
        return marginal + conditional
        
    def dEh(s,h):
        '''
        Relative energy of hidden state compared to lowest energy
        observed in sample
        '''
        return s.Eh_unnormalized(h) - s.E0hs
        
    def dEv(s,v):
        '''
        Relative energy of visible state compared to lowest energy
        observed in sample
        '''
        return s.Ev_unnormalized(v) - s.E0vs
        
    def get_Eh(s,h):
        '''
        '''
        return -(h.dot(s.Bh)+np.sum(np.log1p(sexp(s.Bv+h.dot(s.W))),axis=-1))

    def get_Ev(s,v):
        '''
        '''
        return -(v.dot(s.Bv)+np.sum(np.log1p(sexp(s.Bh+v.dot(s.W.T))),axis=-1))

    def get_dEh(s,h):
        '''
        '''
        E = s.get_Eh(h)
        return E - np.min(E)

    def get_dEv(s,v):
        '''
        '''
        E = s.get_Ev(v)
        return E - np.min(E)

    def get_Hhhv(s,v):
        '''
        '''
        # get conditional firing probabilities
        p = f(s.Bh+v.dot(s.W.T))
        q = 1-p
        return np.sum(-(p*slog(p)+q*slog(q)),axis=-1)

    def get_Hvcondh(s,h):
        '''
        '''
        # get conditional firing probabilities
        p = f(s.Bv+h.dot(s.W))
        q = 1-p
        return np.sum(-(p*slog(p)+q*slog(q)),axis=-1)

    def get_Ehhv_factorial(s,v):
        '''
        '''
        Ph = np.mean(s.Sh,axis=0)
        p = f(s.Bh+v.dot(s.W.T))
        return np.sum(-(p*slog(Ph)+(1-p)*slog(1-Ph)),axis=-1)

    def get_Ehhv_meanfield(s,v):
        '''
        '''
        a = s.Bh+v.dot(s.W.T)
        p = f(a)
        E = -(p.dot(s.Bh)+np.sum(slog(1+sexp(s.Bv+p.dot(s.W))),axis=-1))
        return E - np.min(E)

    def get_Evh(s):
        '''
        '''
        v = s.Sv.T
        a = s.Avh(s.Sh.T)
        logp = -np.sum(v*a-np.log1p(sexp(a)),axis=axis)
        return logp
        
        
    #######################################################################
    # Properties
    
    # ---------------------------------------------------------------------
    # Conditional entropies
    
    @property
    @memoize
    def Hhv(s):
        '''
        Entropy of hiddens conditioned on visible pattern v
        '''
        return np.sum(rb.bernoulli_entropy_activation(s.Ahv(s.Uv.T)),axis=0)
        
    @property
    @memoize
    def barHhv(s):
        '''
        Conditional entropy H(h|v)
        '''
        return np.sum(s.Qv*s.Hhv)
        
    @property
    @memoize
    def Hvh(s):
        '''
        Entropy of visibles conditioned on hidden pattern v
        '''
        return np.sum(rb.bernoulli_entropy_activation(s.Avh(s.Uh.T)),axis=0)
        
    @property
    @memoize
    def barHvh(s):
        '''
        Conditional entropy H(v|h)
        '''
        return np.sum(s.Qh*s.Hvh)
    
    # ---------------------------------------------------------------------
    # Energies of samples
    
    @property
    @memoize
    def EUh_factorial(s):
        '''
        Theoretical energy of unique hidden patterns if it is the case
        that the hidden layer factorizes.
        '''
        return -rb.lnPr_activation(s.Uh,s.Ah,axis=1)
        
    @property
    @memoize
    def dEUh(s):
        '''
        Relative energies for unique hidden patterns in sample, computed
        from model parameters.
        '''
        return s.dEh(s.Uh)
        
    @property
    @memoize
    def dEUv(s):
        '''
        Relative energies for unique visible patterns in sample, computed
        from model parameters.
        '''
        return s.dEv(s.Uv)
    
    # ---------------------------------------------------------------------
    # Conditional energies
        
        
    @property
    @memoize
    def dEhhv_meanfield(s):
        '''
        Compute expected energy by calculating energy of expectation
        '''
        v = s.Uv
        a = s.Bh+v.dot(s.W.T)
        p = f(a)
        E = -(p.dot(s.Bh)+np.sum(g(s.Bv+p.dot(s.W)),axis=-1))
        return E - s.E0hs
        
    @property
    @memoize
    def dEhhv(s,useprior=True):
        '''
        Relative energy of hiddens conditioned on unique observed 
        visible patterns. Computed from visible samples using energies
        from model parameters.
        '''
        v2h = conditional_map(s.Svi,s.dEUh[s.Suidxh])
        E = np.array([np.mean(v2h[vi]) for vi in s.Uvi])
        if not useprior: return E
        K = np.array([np.size(v2h[vi]) for vi in s.Uvi])
        return (E*K+s.dEhhv_meanfield)/(K+1)
        
    @property
    @memoize
    def dEvvh(s,useprior=True):
        '''
        Relative energy of hiddens conditioned on unique observed 
        visible patterns.
        '''
        h2v = conditional_map(s.Shi,s.dEUv[s.Suidxv])
        E = np.array([np.mean(h2v[hi]) for hi in s.Uhi])
     
    @property
    @memoize
    def Evvh(s):
        return s.get_Ehhv_sampled(s.Uv)
    
    @property
    @memoize
    def Evh(s):
        '''
        Conditional energy of visible patterns in sample given corresponding
        hidden pattern
        '''
        return -rb.lnPr_activation(s.Sv.T,s.Avh(s.Sh.T),axis=0)
        
    @property
    @memoize
    def Ehv(s):
        '''
        Conditional energy of hidden patterns in sample given corresponding
        visible pattern
        '''
        return -rb.lnPr_activation(s.Sh.T,s.Ahv(s.Sv.T),axis=0)
    
    @property
    @memoize
    def Evhhv_meanfield(s):
        '''
        Two-hop conditional energies, computed from mean-field
        visibles -> hiddens -> visibles
        '''
        return -rb.lnPr_activation(s.Uv,s.Avh(s.Phv(s.Uv.T)).T)
    
    @property
    @memoize
    def Evhhv(s,useprior=True):
        '''
        Two-hop conditional energies, computed from sample
        visibles -> hiddens -> visibles
        '''
        v2h = conditional_map(s.Svi,s.Evh)
        E = np.array([np.mean(v2h[vi]) for vi in s.Uvi])
        if not useprior: return E
        K = np.array([np.size(v2h[vi]) for vi in s.Uvi])
        return (E*K+s.Evhhv_meanfield)/(K+1)
        
    @property
    @memoize
    def Ehvvh_meanfield(s):
        '''
        Two-hop conditional energies, computed from mean-field
        hiddens -> visibles -> hiddens
        '''
        return -rb.lnPr_activation(s.Uh,s.Ahv(s.Pvh(s.Uh.T)).T)
        
    @property
    @memoize
    def Ehvvh(s,useprior=True):
        '''
        Two-hop conditional energies, computed from sample
        hiddens -> visibles -> hiddens
        '''
        h2v = conditional_map(s.Shi,s.Ehv)
        E = np.array([np.mean(h2v[hi]) for hi in s.Uhi])
        if not useprior: return E
        K = np.array([np.size(h2v[hi]) for hi in s.Uhi])
        return (E*K+s.Ehvvh_meanfield)/(K+1)
        
    #######################################################################
    # Report logging functions
        
    def short_report(s):
        Hhs = np.sum(rb.bitent(s.Ph))
        Hvs = np.sum(rb.bitent(s.Pv))
        # print a short report
        print('\nRBM dataset Ns=%s Nh=%s Nv=%s'%(s.Ns,s.Nh,s.Nv))
        print('Vis capacity, maximum',np.sum(rb.bitent(0.5*np.ones(s.Nv))))
        print('Hid capacity, maximum',np.sum(rb.bitent(0.5*np.ones(s.Nh))))
        print('Vis entropy , sampled',Hvs)
        print('Hid entropy , sampled',Hhs)
        print('Entropy difference   ',(Hhs-Hvs))
        print('Mean hidden rate     ',np.mean(s.Ph))
        print('Mean hidde complexity',rb.bitent(np.mean(s.Ph))*s.Nh)
        
    def long_report(s):
        lgE = np.log2(np.e)
        # Long report
        # print('\nFound dataset %s T=%s Nh=%s Nv=%s'%(DIR,T,Nh,Nv))
        # print('DKL                   %0.2f'%DKL)
        print('\nRBM dataset Ns=%s Nh=%s Nv=%s'%(s.Ns,s.Nh,s.Nv))
        # Hidden layer entropy
        print('==Hidden layer entropy==')
        print('Hid capacity, maximum %0.2f'%(np.sum(rb.bitent(0.5*np.ones(s.Nh)))))
        print('Hid entropy , sampled %0.2f'%(s.Hhs))
        print('Entropy hid sample is %0.2f'%(entropy(s.Qh,base=2)))
        print('<<Eh>h|v>v sampled is %0.2f'%(s.barEhhv*lgE))
        print('<<Eh>h|v>v ufield  is %0.2f'%(s.barEhhv_meanfield*lgE))
        print('Mean hidde complexity %0.2f'%(rb.bitent(np.mean(s.Ph))*s.Nh))
        print('Mean hidden rate      %0.2f'%(np.mean(s.Ph)))
        # Conditional entropy
        print('==Conditional entropy==')
        print('Entropy difference    %0.2f'%(s.Hhs-s.Hvs))
        print('<H_h|v>v           is %0.2f'%(s.barHhv*lgE))
        # Likelihoods
        print('==Negative log-likelihood==')
        print('<<Ev|h>h|v>v sampl is %0.2f'%(s.barEvhhv *lgE))
        print('<<Ev|h>h|v>v ufild is %0.2f'%(s.barEvhhv_meanfield*lgE))
        # KL divergences
        print('==KL divergences==')
        print('<Dkl(h|v||h)>v sam is %0.2f'%(s.barDKLhv*lgE))
        print('<Dkl(h|v||h)>v uf1 is %0.2f'%(s.barDKLhv_meanfield*lgE))
        # Visible entropy; These should be close in value
        print('==Visible layer entropy==')
        print('Vis capacity, maximum %0.2f'%(np.sum(rb.bitent(0.5*np.ones(s.Nv)))))
        print('Vis entropy , sampled %0.2f'%(s.Hvs))
        print('Entropy vis sample is %0.2f'%(entropy(s.Qv,base=2)))
        print('<D(.)+<Ev|h>h|v>v sam %0.2f'%(s.barDKLhv*lgE+s.barEvhhv *lgE))
        print('<D(.)+<Ev|h>h|v>v uf1 %0.2f'%(s.barDKLhv_meanfield*lgE+s.barEvhhv_meanfield*lgE))
