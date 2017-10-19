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
TODO: fix imports here

Examine spiking correlations. time-domain implementation
build a correlation matrix out of cross-correlation estimates
still slow and memory intensive, but not prohibitive
(see efficient solution in frequency domain)
'''

import numpy as np

def ccor(i,j,spikes):
    '''
    TODO: documentation
    
    Cross correlate spikes
    
    Parameters
    ----------
    i : first neuron index 
    j : second neuron index
    spikes : nreplicas x nneurons x samples array of spiking data
    
    Returns
    -------
    x : 
    '''
    A = spikes[:,i,:]
    B = spikes[:,j,:]
    A = A-np.mean(A)
    B = B-np.mean(B)
    x = sum([np.convolve(a,b[::-1],'full')  for (a,b) in zip(A,B)],0)
    return x

def ccm(i,j,k,spikes):
    '''
    TODO: documentation
    
    Construct size k cross-correlation matrix.
    NTrials,NNeurons,NSamples = np.shape(spikes)
    
    Parameters
    ----------
    i : first neuron index 
    j : second neuron index
    spikes : nreplicas x nneurons x samples array of spiking data
    k : TODO
    
    Returns
    -------
    '''
    x = ccor(i,j,spikes)
    midpoint = len(x)//2
    result = np.float64(np.zeros((k,k)))
    for i in np.arange(k):
        result[i,:] = x[midpoint-i:midpoint+k-i]
    return result

def blockccm(k,spikes):
    '''
    TODO: documentation
    
    Generate covariance matrix for linear least squares. It is a block
    matrix of all pairwise cross-correlation matrices
    NTrials,NNeurons,NSamples = np.shape(spikes)
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    NTrials,NNeurons,NSamples = np.shape(spikes)
    result = np.float64(np.zeros((k*NNeurons,)*2))
    for i in np.arange(NNeurons):
        # handle autocorrelation as a special case
        result[i*k:,i*k:][:k,:k] = ccm(i,i,k,spikes)
        for j in np.arange(i):
            # cross-correlations are related by transpose
            cc = ccm(i,j,k,spikes)
            result[i*k:,j*k:][:k,:k] = cc 
            result[j*k:,i*k:][:k,:k] = cc.T
    return result

def sta(i,spikes,lfp):
    '''
    Construct size k STA
    NTrials,NNeurons,NSamples = np.shape(spikes)
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    A = spikes[:,i,:]
    B = lfp
    A = A-np.mean(A)
    B = B-np.mean(B)
    x = np.sum([np.convolve(a,b[::-1],'full') for (a,b) in zip(A,B)],0)
    return x

def blocksta(k,spikes,lfp):
    '''
    TODO: documentation
    
    Block spike-triggered average vector for time-domain least squares
    filter
    NTrials,NNeurons,NSamples = np.shape(spikes)
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    NTrials,NNeurons,NSamples = np.shape(spikes)
    B = np.zeros((k*NNeurons,),dtype=np.float64)
    for i in np.arange(NNeurons):
        x = sta(i,spikes,lfp)
        B[i*k:][:k] = x[len(x)//2-k:][:k]
    return B
    
def reconstruct(k,B,spikes):
    '''
    TODO: documentation
    
    Reconstructs LFP from spikes
    NTrials,NNeurons,NSamples = np.shape(spikes)
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    NTrials,NNeurons,NSamples = np.shape(spikes)
    result = np.zeros((NTrials,NSamples),dtype=np.float64)
    for i in np.arange(NNeurons):
        filt = B[i*k:][:k]
        result += np.array([np.convolve(filt,x,'same') for x in spikes[:,i,:]])
    return result

# frequency domain solution -- confirm that this equals the time-domain
# solution.
# Procedure: operate on each frequency separately
# get cross-spectral matrix for spiking data
# compute least-squares estimator in frequency domain

def cspect(i,j,spikes):
    '''
    Get cross-spectral density as FT of cross-correlation
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    x = ccor(i,j,spikes)
    return np.fft(x)[:len(x)//2+1]

def cspectm(spikes):
    '''
    TODO: documentation
    
    Get all pairs cross spectral matrix
    NTrials,NNeurons,NSamples = np.shape(spikes)
    This is doing much more work than is needed, should change it to
    frequency domain.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    NTrials,NNeurons,NSamples = np.shape(spikes)
    window = hanning(NSamples)
    # precompute fourier transforms
    spikemean = np.mean(spikes,(0,2))
    fts = np.array([[fft(window*(spikes[t,i]-np.mean(spikes[t,i]))) for i in np.arange(NNeurons)] for t in np.arange(NTrials)])
    # compute cross spectra
    result = np.zeros((NSamples,NNeurons,NNeurons),dtype=np.complex128)
    for i in np.arange(NNeurons):
        cs = np.mean([np.conj(trial[i])*trial[i] for trial in fts],0)
        result[:,i,i] = cs
        for j in np.arange(i):
            cs = np.mean([np.conj(trial[i])*trial[j] for trial in fts],0)
            result[:,i,j] = cs
            result[:,j,i] = np.conj(cs)
    return result

def spike_lfp_filters(spikes,lfp):
    '''
    TODO: documentation
    
    Cross-spectral densities between spikes and LFP
    NTrials,NNeurons,NSamples = np.shape(spikes)
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    NTrials,NNeurons,NSamples = np.shape(spikes)
    # precomute lfp fft
    window = hanning(NSamples)
    lfpmean   = np.mean(lfp)
    spikemean = np.mean(spikes,(0,2))
    fftlfp  = [fft((lfp[t]-np.mean(lfp[t]))*window) for t in np.arange(NTrials)]
    result = np.zeros((NSamples,NNeurons),dtype=np.complex128)
    for i in np.arange(NNeurons):
        cspectra = [np.conj(fft((spikes[t,i,:]-np.mean(spikes[t,i,:]))*window))*fftlfp[t] for t in np.arange(NTrials)]
        result[:,i] = np.mean(cspectra,0)
    return result
    
def spectreconstruct(k,B,spikes=None,fftspikes=None):
    '''
    TODO: documentation
    
    Reconstructs LFP from spikes using cross-spectral matrix.
    Can optionally pass the fts if they are already available
    NTrials,NNeurons,NSamples = np.shape(spikes)
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if spikes!=None:
        NTrials,NNeurons,NSamples = np.shape(spikes)
    else:
        NTrials,NNeurons,NSamples = np.shape(fftspikes)
    if ffts==None:
        assert spikes!=None
        fftspikes = np.array([[fft(trial[i]-np.mean(trial[i])) for i in np.arange(NNeurons)] for trial in spikes])
    result = [ifft(sum(fftspikes[t]*B.T,0)) for t in np.arange(NTrials)]
    return result

def create_spectral_model(spikes,lfp,shrinkage=0):
    '''
    TODO: documentation
    
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    XTX = cspectm(spikes)
    XTY = spike_lfp_filters(spikes,lfp)
    shrinkage = eye(np.shape(XTY)[1])*shrinkage
    shrinkage = np.dot(shrinkage.T,shrinkage)
    B = np.array([np.dot(inv(xtx+shrinkage),xty) for xtx,xty in zip(XTX,XTY)])
    return B

def construct_lowpass_operator(fb,k,Fs=1000.0):
    '''
    TODO: documentation
    
    Constructs a low-pass regularization operator
    Get the impulse response of a low-pass filter first
    Then copy it into the matrix.
    This really only makes sense in the time domain.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    ff = np.zeros((k*4),dtype=np.float64)
    ff[k*2]=1
    ff = bandfilter(ff,fb=fb,Fs=Fs)
    result = np.zeros((k,k),dtype=np.float64)
    for i in np.arange(k):
        result[i] = ff[k*2-i:][:k]
    return result

def autocorrelation_bayes(s,D=200,prior_var=None):
    '''
    Computes autocorrelation of signal `s` over time lags `D`, 
    applying a Gaussian prior of mean zero and variance `prior_var`.
    If `prior_var` is None, then the length of signal `s` is used.
    
    Parameters
    ----------
    s : sequence of values
    D : number of lags to compute, default is 200
    prior_var : positive scalar or None, default is None
    
    Returns
    -------
    xc: autocorrelation over lags D, with zero-lag variance
    '''
    xc   = np.zeros(D+1)
    lags = np.arange(0,D+1)
    for lag in lags:
        a = s[lag:]
        b = s[:len(s)-lag]
        a = a-np.mean(a)
        b = b-np.mean(b)
        cm = np.mean(a*b)
        n = len(a)
        cv = np.var(a*b)*n/(n-1)
        cm = cm / (1 + cv/len(s))
        xc[lag] = cm
    return xc


'''
# simulated sanity check
k = 30
N = 2000
sim_spikes = rand(N)<0.04
sim_sta = np.zeros((k,),dtype=np.float64)
sim_sta[k//2:]=1
sim_lfp = convolve(sim_sta,sim_spikes,'same')
clf()
plot(sim_lfp,'m')
plot(sim_spikes,'c')
sim_spikes = np.array([[sim_spikes]])
sim_lfp = np.array([sim_lfp])
XTX=blockccm(k,sim_spikes)
XTY=blocksta(k,sim_spikes,sim_lfp)
# linear least squares model estimate
B = np.dot(pinv(XTX),XTY)
# Check time-domain solution: did it work?
original = ravel(sim_lfp)
reconstruction = ravel(reconstruct(k,B,sim_spikes))
RMSE = sqrt(np.mean((original-reconstruction)**2))
print 'RMSE = ',RMSE
plot(original,'b')
plot(reconstruction,'g')
'''


'''
# simulated sanity check
k = 30
N = 3000
sim_spikes = rand(N)<0.005
sim_sta = np.zeros((k,),dtype=np.float64)
sim_sta[k//2:]=1
sim_lfp = convolve(sim_sta,sim_spikes,'same')
clf()
plot(sim_lfp,'m')
plot(sim_spikes,'c')
sim_spikes = np.array([[sim_spikes]])
sim_lfp = np.array([sim_lfp])
XTX = cspectm(sim_spikes)
XTY = spike_lfp_filters(sim_spikes,sim_lfp)
# solve each frequency separately
B = np.array([np.dot(pinv(xtx),xty) for xtx,xty in zip(XTX,XTY)])
# B should be something like the cross-spectral densirt between the 
# spiking population and LFP now
# sort of a population coherence with the LFP? 
# plot(fftfreq(len(B)*2,BINSIZE/float(Fs))[:len(B)],np.mean(abs(B),1))
reconstruction = spectreconstruct(k,B,sim_spikes)
plot(ravel(reconstruction),'y')
plot(ravel(sim_lfp),'c')
'''

'''
# cross-validated simulated sanity check with regularization
k = 30
N = 3000
sim_sta = np.zeros((k,),dtype=np.float64)
sim_sta[k//2:]=1
NTrials = 2
sim_spikes = [rand(N)<0.05 for i in np.arange(NTrials)]
sim_lfp    = [convolve(sim_sta,sim_spikes[i],'same') for i in np.arange(NTrials)]
spikes = np.array(sim_spikes)[:,None,:]
lfp    = sim_lfp
lfp   -= np.mean(lfp)
clf()
for shrinkage in np.arange(20):
    B = create_spectral_model(spikes[:NTrials//2],lfp[:NTrials//2],shrinkage=shrinkage)
    original       = ravel(lfp[NTrials//2:])
    reconstruction = ravel(spectreconstruct(k,B,spikes[NTrials//2:]))
    plot(original,'c')
    plot(reconstruction,color=(shrinkage/20.0,)*3)
    RMSE = sqrt(np.mean((original-reconstruction)**2))
    print 'shrinkage = ',shrinkage
    print 'RMSE = ',RMSE
'''

