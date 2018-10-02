#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
"""
Routines for calculating coherence
"""

from numpy import *
import numpy as np
from collections import defaultdict

from multiprocessing import cpu_count
__N_CPU__ = cpu_count()

import scipy.stats

from neurotools.signal.morlet     import *
from neurotools.getfftw           import *
from neurotools.signal     import zscore
from neurotools.signal.multitaper import dpss_cached
from neurotools.stats.circular    import squared_first_circular_moment
try:
    import nitime
    from nitime.algorithms import coherence
except:
    print('could not locate nitime module; coherence functions missing')
    def coherence(*args,**kwargs):
        raise ImportError("nitime module not loaded, coherence missing")

def morlet_population_synchrony_spectrum(lfp,fa,fb,w=4.0,resolution=0.1,Fs=1000):
    '''
    First dimension is nchannels, second is time.
    Use morlet wavelets ( essentially bandpass filter bank ) to compute
    short-timescale synchrony.
    for each band: take morlet spectrum over time.
    take kuromoto or synchrony measure over complex vectors attained
    '''
    freqs, transformed = fft_cwt(lfp.T,fa,fb,w,resolution,Fs)
    return freqs, abs(mean(transformed,0))/mean(abs(transformed),0)

def population_eigencoherence(lfp,fa,fb,w=4.0,resolution=0.1,Fs=1000):
    '''
    Uses the eigenvalue spectrum of the pairwise coherence matrix.
    In the case of wavelets, each time-frequency point has one
    complex value.
    The matrix we build will be I think $|z_i z_j|$
    ... this will involve a lot of computation.
    ... let's not do it.
    See ramirez et al
    A GENERALIZATION OF THE MAGNITUDE SQUARED COHERENCE SPECTRUM FOR
    MORE THAN TWO SIGNALS: DEFINITION, PROPERTIES AND ESTIMATION
    '''
    raise NotImplementedError()

def population_coherence_matrix(lfp):
    '''
    lfp is a Nch×NTime matrix of data channels.
    ntapers is a positive integer.
    For each pair of channels compute multitaper coherence.
    take the product of each taper with each channel and take the FT
    '''
    NCH,N = shape(lfp)
    tapers,eigen = dpss_cached(N,10.0)
    M = sum(eigen)**2/sum(eigen**2) # adjusted sample size
    tapered = arr([fft(lfp*taper,axis=1) for taper in tapers])
    NT = len(eigen)
    def magsq(z):
        return real(z*conj(z))
    psd = arr([sum([magsq(tapered[k,i,:])*eigen[k] for k in range(NT)],0)/sum(eigen) for i in range(NCH)])
    results = zeros((NCH,NCH,N),dtype=float64)
    for i in range(NCH):
        results[i,i]=1
        for j in range(i):
            a = tapered[:,i,:]
            b = tapered[:,j,:]
            nn = sum([a[k]*conj(b[k])*eigen[k] for k in range(NT)],0)/sum(eigen)
            sqrcoherence = sqrt(magsq(nn)/(psd[i]*psd[j]))
            unbiased = (M*sqrcoherence-1)/(M-1)
            results[i,j,:]=results[j,i,:]=unbiased
    factored = zeros((NCH,NCH,N),dtype=float64)
    spectra  = zeros((NCH,N),dtype=float64)
    for i in range(N):
        w,v = eig(results[:,:,i])
        v[:,w<0]*=-1
        w[w<0]*=-1
        order = argsort(w)
        spectra [:,i]   = w[order]
        factored[:,:,i] = v[:,order]
    freqs = fftfreq(N,1./1000)
    return freqs,results,spectra,factored

def multitaper_coherence(x,y,Fs=1000,BW=5):
    '''
    multitaper_coherence(x,y,Fs=1000,BW=5)
    BW is the multitaper bandwidth
    returns freqs, cohere
    '''
    x -= mean(x)
    y -= mean(y)
    method = {'this_method':'multi_taper_csd','BW':BW,'Fs':Fs}
    freqs,cohere = coherence(np.array([x,y]),method)
    N = len(x)
    freqs = abs(fftfreq(N,1./Fs)[:N/2+1])
    return freqs, cohere[0,1]

def sliding_multitaper_coherence(x,y,window=500,step=100,Fs=1000,BW=5):
    '''
    Sliding multitaper coherence between x and y
    This is a somewhat strange implementation that is only preserved for
    legacy reasons.
    '''
    N = len(x)
    assert len(y)==N
    allcohere = []
    for tstart in xrange(0,N-window+1,step):
        ff,cohere = multitaper_coherence(x[tstart:tstart+window],y[tstart:tstart+window],Fs,BW)
        allcohere.append(cohere)
    return ff,np.array(allcohere)

def sliding_multitaper_coherence_parallel(x,y,window=500,step=100,Fs=1000,BW=5):
    '''
    Sliding multitaper coherence between x and y
    Takes multiple samples over time, but estimates each sample using multi-taper
    See also multitaper_coherence
    This is a somewhat strange implementation that is only preserved for
    legacy reasons.
    '''
    N = len(x)
    assert len(y)==N
    allcohere = []
    problems = [(tstart,(x[tstart:tstart+window],y[tstart:tstart+window],Fs,BW)) for tstart in xrange(0,N-window+1,step)]
    allcohere = squeeze(np.array(parmap(mtmchpar,problems)))
    freqs = abs(fftfreq(window,1./Fs)[:window/2+1])
    return freqs,allcohere

def coherence_pvalue(C,NSample,beta = 23/20.):
    '''
    Jarvis & Mitra (Neural Comp., 2001, p732)
    Pesaran et al. (Nature, 2008, supp info, p5)

    beta = 23/20. Jarvis & Mitra suggest (Neural Comp., 2001, p732)
    Pesaran et al. suggest beta=1.5 (Nature, 2008, supp info, p5)

    \citep{jarvis2001sampling, pesaran2008free}
    '''
    df = 2*NSample                 # degrees of freedom
    q  = sqrt(-(df-2)*log(1-C**2)) # ???
    Z  = beta*(q-beta)             # z-transformed coherence
    # testing whether coherence is greater than zero:
    # to get the chance level p-value, use the right tail, i.e. 1 - cdf
    # here the p-value is computed based both on the t and normal distributions, respectively
    return 1-scipy.stats.t.cdf(Z,df)

def multitaper_multitrial_coherence(x,
    Fs            = 1000,
    NTapers       = 5,
    test          = 'pvalue',
    NRandomSample = 100,
    unbiased      = False,
    eps           = 1e-30,
    parallel      = True):
    '''
    multitaper_multitrial_coherence(x,y,Fs=1000,NT=5)
    Computes coherence over multiple tapers and multiple trials

    x: data. NVariables×NTrials×NTime

    NTapers: number of tapers, defaults to 5

    bootstrap: defaults to 100
        If bootstrap is a positive integer, we will perform bootstrap
        resampling and return this distribution along with the coherence
        result.

    unbiased: defaults to True
        If true it will apply the standard bias correction for averaging
        of circular data, which should remove sample-size dependence for
        the coherence values, at the cost of increasing estimator variance
        and occassionally generating strange (negative) coherence values.
        Bias correction for magnitude squared is (N|z|²-1)/(N-1)

    Procedure:
    1   Z-score each trial (removes mean)
    2   Generate tapers
    3   Compute tapered FFTs for all trials and tapers
    4   Cross-spectral density

    return freqs, coherence, bootstrapped
    '''
    if not len(shape(x)) == 3:
        raise ValueError("Expected 3 dimensional data: vars x trails x times")
    if not test in (None,'bootstrap','shuffle','pvalue'):
        raise ValueError("test must be None, bootstrap, shuffle, pvalue")
    x = zscore(np.array(x).T,axis=0).T

    NVar, NTrials, NTime = shape(x)
    NSample = NTapers * NTrials

    tapers, eigen = dpss_cached(NTime,0.4999*NTapers)

    NThread = max(1,__N_CPU__-1) if parallel else 1

    FMax  = NTime//2+1
    freqs = abs(fftfreq(NTime,1./Fs)[:FMax]) # NFreq
    ft = np.array([fft(x*t,axis=-1,threads=NThread)\
            for t in tapers])                # NTapers×NVar×NTrials×NTime
    ft = np.swapaxes(ft,0,1)                 # NVar×NTapers×NTrials×NTime
    ft = ft.reshape((NVar,NSample,NTime))    # NVar×NSample×NTime
    ft = ft[:,:,:FMax]                       # NVar×NSample×NFreq
    if (NTime%2==0): ft[:,:,1:-1]*=2         # NVar×NSample×NFreq
    else:            ft[:,:,1:  ]*=2         # NVar×NSample×NFreq
    psds = (abs(ft)**2)                      # NVar×NSample×NFreq
    psd  = np.mean(psds,axis=1)              # NVar×NFreq

    def _coherence_(pij,pii,pjj):
        return (pij+eps)/(pii*pjj+eps)

    coherence = {}
    for i in range(NVar):
        coherence[i] = psd[i]
        for j in range(i):
            pij = squared_first_circular_moment(\
                np.conj(ft[i])*ft[j],
                axis=0,
                unbiased=unbiased)
            coherence[i,j] = coherence[j,i] = _coherence_(pij,psd[i],psd[j])

    if test is None:
        samples = None

    elif test=='pvalue':
        # Use transformed null distribution to estiamte a z-score, then use
        # a t-test.
        samples = {}
        for i in range(NVar):
            for j in range(i):
                samples[i,j] = samples[j,i] = coherence_pvalue(coherence[i,j],NSample)
    else:
        # sample with replacement bootstrap number of times
        # we do this by resampling from trials/tapers
        if test=='bootstrap':
            samples = defaultdict(list)
            for bi in xrange(NRandomSample):
                sampled = np.random.choice(NSample,NSample)
                dof     = len(np.unique(sampled)) # this correction is not enough, but better than nothing
                bpsd    = np.mean(psds[:,sampled],axis=0)
                for i in range(NVar):
                    for j in range(i):
                        pij = squared_first_circular_moment(\
                            np.conj(ft[i,sampled])*ft[j,sampled],
                            axis=0,
                            unbiased=unbiased,
                            dof=dof)
                        samples[i,j].append( _coherence_(pij,bpsd[i],bpsd[j]))

        # trial-shuffling estimate of chance level
        elif test=='shuffle':
            samples = defaultdict(list)
            for si in xrange(NRandomSample):
                perms = [np.random.permutation(NSample) for i in xrange(NVar)]
                for i in range(NVar):
                    for j in range(i):
                        pij = squared_first_circular_moment(\
                            np.conj(ft[i,perms[i]])*ft[j,perms[j]],
                            axis=0,
                            unbiased=unbiased)
                        samples[i,j].append( _coherence_(pij,psd[i],psd[j]))

        # Sort samples and convert to regular dictionary
        for i in range(NVar):
            for j in range(i):
                b = np.array(samples[i,j])
                b.sort(axis=0)
                samples[i,j] = samples[j,i] = b
        samples = dict(samples)

    return freqs, coherence, samples



if __name__=="__main__":
    print("Testing coherence code on CGID data")
    from cgid.setup import *
    okwarn()
    session = 'RUS120518'
    area = 'M1'
    trial = get_good_trials(session,area)[0]
    csd = []
    for i in range(0,7000-100,5):
        print(i)
        lfp = get_all_lfp(session,area,trial,(6,-1000+i,-1000+i+100))
        s = population_coherence_matrix(lfp)[2][0]
        csd.append(s)
    # Coherence example
    Fs = 1000
    N = Fs*4
    t = arange(N)
    s1 = cos(t*2*pi*90/Fs)
    s2 = cos(t*2*pi*40/Fs)
    n1 = randn(N)
    n2 = randn(N)
    n3 = randn(N)
    g1 = n1+n2
    g2 = n3+n2
    tapers,eigen = dpss_cached(N,10.0)
    tapered1 = np.array([fft(g1*taper,axis=1) for taper in tapers])
    tapered2 = np.array([fft(g2*taper,axis=1) for taper in tapers])
    DF = sum(eigen)
    NT = len(eigen)
    psd1 = sum([magsq(tapered1[k,:])*eigen[k] for k in range(NT)],0)/DF
    psd2 = sum([magsq(tapered2[k,:])*eigen[k] for k in range(NT)],0)/DF
    def magsq(z):
        return real(z*conj(z))
    nn = sum([tapered1[k]*tapered2(b[k])*eigen[k] for k in range(NT)],0)/DF
    coherence = magsq(nn)/(psd[i]*psd[j])
    plot(coherence)
