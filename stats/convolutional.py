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

# examine spiking correlations. time-domain implementation
# build a correlation matrix out of cross-correlation estimates
# still slow and memory intensive, but not prohibitive
# (see efficient solution in frequency domain)

def ccor(i,j,spikes):
    '''
    Cross correlate spikes
    '''
    A = spikes[:,i,:]
    B = spikes[:,j,:]
    A = A-mean(A)
    B = B-mean(B)
    x = sum([convolve(a,b[::-1],'full')  for (a,b) in zip(A,B)],0)
    return x

def ccm(i,j,k,spikes):
    '''
    Construct size k cross-correlation matrix.
    NTrials,NNeurons,NSamples = shape(spikes)
    '''
    x = ccor(i,j,spikes)
    midpoint = len(x)//2
    result = float64(zeros((k,k)))
    for i in range(k):
        result[i,:] = x[midpoint-i:midpoint+k-i]
    return result

def blockccm(k,spikes):
    '''
    Generate covariance matrix for linear least squares. It is a block
    matrix of all pairwise cross-correlation matrices
    NTrials,NNeurons,NSamples = shape(spikes)
    '''
    NTrials,NNeurons,NSamples = shape(spikes)
    result = float64(zeros((k*NNeurons,)*2))
    for i in range(NNeurons):
        # handle autocorrelation as a special case
        result[i*k:,i*k:][:k,:k] = ccm(i,i,k,spikes)
        for j in range(i):
            # cross-correlations are related by transpose
            cc = ccm(i,j,k,spikes)
            result[i*k:,j*k:][:k,:k] = cc 
            result[j*k:,i*k:][:k,:k] = cc.T
    return result

def sta(i,spikes,lfp):
    '''
    Construct size k STA
    NTrials,NNeurons,NSamples = shape(spikes)
    '''
    A = spikes[:,i,:]
    B = lfp
    A = A-mean(A)
    B = B-mean(B)
    x = sum([convolve(a,b[::-1],'full') for (a,b) in zip(A,B)],0)
    return x

def blocksta(k,spikes,lfp):
    '''
    Block spike-triggered average vector for time-domain least squares
    filter
    NTrials,NNeurons,NSamples = shape(spikes)
    '''
    NTrials,NNeurons,NSamples = shape(spikes)
    B = zeros((k*NNeurons,),dtype=float64)
    for i in range(NNeurons):
        x = sta(i,spikes,lfp)
        B[i*k:][:k] = x[len(x)//2-k:][:k]
    return B
    
def reconstruct(k,B,spikes):
    '''
    Reconstructs LFP from spikes
    NTrials,NNeurons,NSamples = shape(spikes)
    '''
    NTrials,NNeurons,NSamples = shape(spikes)
    result = zeros((NTrials,NSamples),dtype=float64)
    for i in range(NNeurons):
        filt = B[i*k:][:k]
        result += array([convolve(filt,x,'same') for x in spikes[:,i,:]])
    return result

# frequency domain solution -- confirm that this equals the time-domain
# solution.
# Procedure: operate on each frequency separately
# get cross-spectral matrix for spiking data
# compute least-squares estimator in frequency domain

def cspect(i,j,spikes):
    '''
    Get cross-spectral density as FT of cross-correlation
    '''
    x = ccor(i,j,spikes)
    return fft(x)[:len(x)//2+1]

def cspectm(spikes):
    '''
    Get all pairs cross spectral matrix
    NTrials,NNeurons,NSamples = shape(spikes)
    This is doing much more work than is needed, should change it to
    frequency domain.
    '''
    NTrials,NNeurons,NSamples = shape(spikes)
    window = hanning(NSamples)
    # precompute fourier transforms
    spikemean = mean(spikes,(0,2))
    fts = array([[fft(window*(spikes[t,i]-mean(spikes[t,i]))) for i in range(NNeurons)] for t in range(NTrials)])
    # compute cross spectra
    result = zeros((NSamples,NNeurons,NNeurons),dtype=complex128)
    for i in range(NNeurons):
        cs = mean([conj(trial[i])*trial[i] for trial in fts],0)
        result[:,i,i] = cs
        for j in range(i):
            cs = mean([conj(trial[i])*trial[j] for trial in fts],0)
            result[:,i,j] = cs
            result[:,j,i] = conj(cs)
    return result

def spike_lfp_filters(spikes,lfp):
    '''
    Cross-spectral densities between spikes and LFP
    NTrials,NNeurons,NSamples = shape(spikes)
    '''
    NTrials,NNeurons,NSamples = shape(spikes)
    # precomute lfp fft
    window = hanning(NSamples)
    lfpmean   = mean(lfp)
    spikemean = mean(spikes,(0,2))
    fftlfp  = [fft((lfp[t]-mean(lfp[t]))*window) for t in range(NTrials)]
    result = zeros((NSamples,NNeurons),dtype=complex128)
    for i in range(NNeurons):
        cspectra = [conj(fft((spikes[t,i,:]-mean(spikes[t,i,:]))*window))*fftlfp[t] for t in range(NTrials)]
        result[:,i] = mean(cspectra,0)
    return result
    
def spectreconstruct(k,B,spikes=None,fftspikes=None):
    '''
    Reconstructs LFP from spikes using cross-spectral matrix.
    Can optionally pass the fts if they are already available
    NTrials,NNeurons,NSamples = shape(spikes)
    '''
    if spikes!=None:
        NTrials,NNeurons,NSamples = shape(spikes)
    else:
        NTrials,NNeurons,NSamples = shape(fftspikes)
    if ffts==None:
        assert spikes!=None
        fftspikes = array([[fft(trial[i]-mean(trial[i])) for i in range(NNeurons)] for trial in spikes])
    result = [ifft(sum(fftspikes[t]*B.T,0)) for t in range(NTrials)]
    return result

def create_spectral_model(spikes,lfp,shrinkage=0):
    XTX = cspectm(spikes)
    XTY = spike_lfp_filters(spikes,lfp)
    shrinkage = eye(shape(XTY)[1])*shrinkage
    shrinkage = dot(shrinkage.T,shrinkage)
    B = array([dot(inv(xtx+shrinkage),xty) for xtx,xty in zip(XTX,XTY)])
    return B

def construct_lowpass_operator(fb,k,Fs=1000.0):
    '''
    Constructs a low-pass regularization operator
    Get the impulse response of a low-pass filter first
    Then copy it into the matrix.
    This really only makes sense in the time domain.
    '''
    ff = zeros((k*4),dtype=float64)
    ff[k*2]=1
    ff = bandfilter(ff,fb=fb,Fs=Fs)
    result = zeros((k,k),dtype=float64)
    for i in range(k):
        result[i] = ff[k*2-i:][:k]
    return result


'''
# simulated sanity check
k = 30
N = 2000
sim_spikes = rand(N)<0.04
sim_sta = zeros((k,),dtype=float64)
sim_sta[k//2:]=1
sim_lfp = convolve(sim_sta,sim_spikes,'same')
clf()
plot(sim_lfp,'m')
plot(sim_spikes,'c')
sim_spikes = array([[sim_spikes]])
sim_lfp = array([sim_lfp])
XTX=blockccm(k,sim_spikes)
XTY=blocksta(k,sim_spikes,sim_lfp)
# linear least squares model estimate
B = dot(pinv(XTX),XTY)
# Check time-domain solution: did it work?
original = ravel(sim_lfp)
reconstruction = ravel(reconstruct(k,B,sim_spikes))
RMSE = sqrt(mean((original-reconstruction)**2))
print 'RMSE = ',RMSE
plot(original,'b')
plot(reconstruction,'g')
'''


'''
# simulated sanity check
k = 30
N = 3000
sim_spikes = rand(N)<0.005
sim_sta = zeros((k,),dtype=float64)
sim_sta[k//2:]=1
sim_lfp = convolve(sim_sta,sim_spikes,'same')
clf()
plot(sim_lfp,'m')
plot(sim_spikes,'c')
sim_spikes = array([[sim_spikes]])
sim_lfp = array([sim_lfp])
XTX = cspectm(sim_spikes)
XTY = spike_lfp_filters(sim_spikes,sim_lfp)
# solve each frequency separately
B = array([dot(pinv(xtx),xty) for xtx,xty in zip(XTX,XTY)])
# B should be something like the cross-spectral densirt between the 
# spiking population and LFP now
# sort of a population coherence with the LFP? 
# plot(fftfreq(len(B)*2,BINSIZE/float(Fs))[:len(B)],mean(abs(B),1))
reconstruction = spectreconstruct(k,B,sim_spikes)
plot(ravel(reconstruction),'y')
plot(ravel(sim_lfp),'c')
'''

'''
# cross-validated simulated sanity check with regularization
k = 30
N = 3000
sim_sta = zeros((k,),dtype=float64)
sim_sta[k//2:]=1
NTrials = 2
sim_spikes = [rand(N)<0.05 for i in range(NTrials)]
sim_lfp    = [convolve(sim_sta,sim_spikes[i],'same') for i in range(NTrials)]
spikes = array(sim_spikes)[:,None,:]
lfp    = sim_lfp
lfp   -= mean(lfp)
clf()
for shrinkage in range(20):
    B = create_spectral_model(spikes[:NTrials//2],lfp[:NTrials//2],shrinkage=shrinkage)
    original       = ravel(lfp[NTrials//2:])
    reconstruction = ravel(spectreconstruct(k,B,spikes[NTrials//2:]))
    plot(original,'c')
    plot(reconstruction,color=(shrinkage/20.0,)*3)
    RMSE = sqrt(mean((original-reconstruction)**2))
    print 'shrinkage = ',shrinkage
    print 'RMSE = ',RMSE
'''

