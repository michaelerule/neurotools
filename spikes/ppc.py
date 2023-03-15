#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Pairwise-phase-consistency spike-LFP coupling statistics
and related functions. 
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

try:
    from spectrum.mtm import dpss
except:
    def dpss(*args):
        raise NotImplementedError(
            "Please install the spectrum module, e.g."
            "\n\tpip install spectrum\n"
            "to use this functionality.")

import types
import numpy as np
from numpy.fft import *
from warnings import warn
import neurotools.signal as sig

__PPC_FP_TYPE__=np.longdouble

def phase_randomize(signal):
    '''
    Phase randomizes a signal by rotating frequency 
    components by a random angle. Negative frequencies are 
    rotated in the opposite direction. The nyquist 
    frequency, if present, has it's sign randomly flipped.
    
    Parameters
    ----------
    signal : 1D array

    Returns
    -------
    phase-randomized sigal
    '''
    assert 1==len(signal.shape)
    N = int(len(signal))
    if N%2==1:
        # signal length is odd.
        # ft will have one DC component then symmetric frequency components
        randomize  = np.exp(1j*np.random.rand((N-1)//2)*2*np.pi)
        conjugates = np.conj(randomize)[::-1]
        randomize  = np.append(randomize,conjugates)
    else:
        # signal length is even
        # will have one single value at the nyquist frequency
        # which will be real and can be sign flipped but not rotated
        flip = 1 if np.random.rand(1)<0.5 else -1
        randomize  = np.exp(1j*np.random.rand((N-2)//2)*2*np.pi)
        conjugates = np.conj(randomize)[::-1]
        randomize  = np.append(randomize,flip)
        randomize  = np.append(randomize,conjugates)
    # the DC component is not randomized
    randomize = np.append(1,randomize)
    # take FFT and apply phase randomization
    ff = np.fft.fft(signal)*randomize
    # take inverse
    randomized = np.fft.ifft(ff)
    return randomized.real


def fftppc_biased(snippits,Fs=1000,taper=None):
    '''
    FFT-based pairwise phase consistency *without*
    corrections for finite-sample-size bias. 

    Parameters
    ----------
    snippits:
        List of LFP signals extracted in the vicinity of
        each spike.
    
    Other Parameters
    ----------------
    Fs: positive int; default 1000
        Sample rate
    taper: np.attay
        Windowing function to apply before taking the FFT        

    Returns
    -------
    freqs: npp.array
        Frequencies at which the PPC has been evaluated
    raw: np.array
        Raw (biased) value for the PPC at each frequency
    phases: np.array
        Phase values associated with each ppc coefficient

    '''
    # some precision trouble
    # use quad precition
    # PPC doesn't care about scale so also rescale?
    snippits = np.array(snippits,dtype=__PPC_FP_TYPE__)
    snippits = snippits/np.std(snippits)
    M,window = np.shape(snippits)
    if not taper is None:
        snippits = snippits*taper;
    if M<=window: warn('WARNING SAMPLES SEEM TRANSPOSED?')
    fs       = np.fft.fft(snippits)
    average  = np.mean(fs/abs(fs),0)
    raw      = np.abs(average)**2
    phases   = np.angle(average)
    freqs    = np.fft.fftfreq(window,1./Fs)
    return freqs[:(window+1)/2], raw[:(window+1)/2], phases[:(window+1)/2]

def fftppc(snippits,Fs=1000,taper=None):
    '''
    FFT-based pairwise phase consistency **with** 
    corrections for finite-sample-size bias. 

    Parameters
    ----------
    snippits:
        List of LFP signals extracted in the vicinity of
        each spike.
    
    Other Parameters
    ----------------
    Fs: positive int; default 1000
        Sample rate
    taper: np.attay
        Windowing function to apply before taking the FFT        

    Returns
    -------
    freqs: npp.array
        Frequencies at which the PPC has been evaluated
    raw: np.array
        Raw (biased) value for the PPC at each frequency
    phases: np.array
        Phase values associated with each ppc coefficient
    '''
    '''
    make sure this is equivalent to the following matlab lines
    raw     = abs(sum(S./abs(S))).^2;
    ppc     = (raw - M)./(M.*(M-1));

    the raw computation uses the mean which includes a 1/M factor
    which is then squared so the raw PPC is already /M^2 compared
    to the PPC value intermediate in the matlab code. So we have to
    multiply it by M^2
    '''
    M,window = shape(snippits)
    ff,raw,phase = fftppc_biased(snippits,Fs,taper=taper)
    unbiased = (raw*M-1)/(M-1)
    return ff, unbiased, phase

def fftppc_biased_multitaper(snippits,Fs=1000,k=4,transpose_warning=True):
    '''
    FFT-based pairwise phase consistency **without** 
    corrections for finite-sample-size bias, using a
    multi-taper method with `k` tapers to reduce variance
    at the expense of bandwidth resolution.  
    
    Parameters
    ----------
    snippits: Nspikes x Nwindow
        Array of spike-triggered samples of the signal trace
    
    Other Parameters
    ----------------
    Fs: scalar
        Sampling frequency. Defaults to 1000 Hz
    k: positive integer
        Number of tapers. Defaults to 4.
    transpose_warning: boolean; default True
        Warn if any of the input arrays appear transposed.

    Returns
    -------
    freqs: npp.array
        Frequencies at which the PPC has been evaluated
    raw: np.array
        Raw (biased) value for the PPC at each frequency
    phases: np.array
        Phase values associated with each ppc coefficient
    '''
    # some precision trouble
    # use quad precition
    # PPC doesn't care about scale so also rescale?
    #nippits = array(snippits,dtype=__PPC_FP_TYPE__)
    #snippits = snippits/std(snippits)
    M,window = np.shape(snippits)
    if transpose_warning and M<=window: warn('WARNING SAMPLES SEEM TRANSPOSED?')
    #print('BROKE FOR MATLAB CHECK REMOVE +1 on K KEEP ALL TAPERS')
    #tapers   = dpss(window,NW=0.499*(k+1),k=(k+1))[0][:,:-1]
    tapers   = dpss(window,NW=0.499*k,k=k)[0]
    results  = []
    unit  = lambda x:x/abs(x)
    average = [np.mean(unit(np.fft.fft(snippits*taper)),0) for taper in tapers.T]
    raw     = np.mean([np.abs(x)**2 for x in average],0)
    phases  = np.angle(np.mean([np.exp(2j*np.pi*np.angle(x)) for x in average],0))
    freqs   = np.fft.fftfreq(window,1./Fs)
    return freqs[:(window+1)//2], raw[:(window+1)//2], phases[:(window+1)//2]

def fftppc_multitaper(snippits,Fs=1000,k=4,transpose_warning=True):
    '''
    FFT-based pairwise phase consistency **with** 
    corrections for finite-sample-size bias, using a
    multi-taper method with `k` tapers to reduce variance
    at the expense of bandwidth resolution.  
    
    Parameters
    ----------
    snippits: Nspikes x Nwindow
        Array of spike-triggered samples of the signal trace
    
    Other Parameters
    ----------------
    Fs: scalar
        Sampling frequency. Defaults to 1000 Hz
    k: positive integer
        Number of tapers. Defaults to 4.
    transpose_warning: boolean; default True
        Warn if any of the input arrays appear transposed.

    Returns
    -------
    freqs: npp.array
        Frequencies at which the PPC has been evaluated
    raw: np.array
        Raw (biased) value for the PPC at each frequency
    phases: np.array
        Phase values associated with each ppc coefficient
    '''
    # some precision trouble
    # use quad precition
    # PPC doesn't care about scale so also rescale?
    #snippits = array(snippits,dtype=__PPC_FP_TYPE__)
    M,window = np.shape(snippits)
    ff,raw,phase = fftppc_biased_multitaper(snippits,Fs,k,transpose_warning)
    unbiased = (raw*k*M-1)/(M-1)
    return ff, unbiased, phase


def discard_spikes_closer_than_delta(signal,times,delta):
    '''
    When computing PPC, we need to throw out spikes that are too close
    together. This is a heuristic to make the spiking samples
    "more independent". We also need to skip spikes that are so close to
    the edge of our data that we can't get the LFP window surrounding
    the spike time. Because certain tests that compare across conditions
    require matching the number of spikes, to ensure that the variance
    of the PPC estimator is comparable between the conditions, we expose
    the code for selecting the subset of spikes for PPC here, so that
    it can be used to ensure that both conditions have a matching number
    of spikes

    Parameters
    ----------
    signal:
    times:
    delta:
    
    
    Returns
    -------
    '''
    N = len(signal)
    times = array(times)
    #print('%d spikes total'%len(times))
    times = times[times>=delta]
    times = times[times<N-delta]
    #print('%d spikes with enough padding'%len(times))
    usetimes = []
    t_o = -inf
    for t in times:
        if t-t_o>=delta:
            t_o = t
            usetimes.append(t)
    #print('%d spikes far enough apart to be usable'%len(usetimes))
    return usetimes

def pairwise_phase_consistancy(
    signal,
    times,
    window=50,
    Fs=1000,
    k=4,
    multitaper=True,
    biased=False,
    delta=100,
    taper=None):
    '''

    Parameters
    ----------
    signal: 
        1D real valued signal
    times:  
        Times of events relative to signal
    
    Other Parameters
    ----------------
    window: positive int; default 50
        Time around event to examine
    Fs: positive int; default 1000
        sample rate for computing freqs
    k: positive int; default 4
        number of tapers
        Also accepts lists of signals / times
        returns (freqs, ppc, phase), lfp_segments
    multitaper: boolean; default True
    biased: booleanl default False
    delta: positive int; default 100
    taper: array; default None
    
    Returns
    -------
    freqs: npp.array
        Frequencies at which the PPC has been evaluated
    raw: np.array
        Raw (biased) value for the PPC at each frequency
    phases: np.array
        Phase values associated with each ppc coefficient

    '''
    if multitaper:
        print(
        "Warning: multitaper can introduce a bias into PPC "
        "that depends on the number of tapers!")
        print(
        "For a fixed number of tapers, the bias is "
        "constant, but be careful")

    if not taper is None and multitaper:
        print(
        "A windowing taper was specified, but multitaper "
        "mode was also selected? The taper argument is for "
        "providing a windowing function when not using "
        "multitaper estimation")
        assert 0
    if type(taper) is types.FunctionType:
        taper = taper(window*2+1)
    if biased: warn('skipping bias correction entirely')
    assert window>0
    if len(shape(signal))==1:
        usetimes = discard_spikes_closer_than_delta(signal,times,delta)
        snippits = array([
            sig.zeromean(signal[t-window:t+window+1]) for t in usetimes
        ])
    elif len(shape(signal))==2:
        warn('assuming first dimension is trials / repititions')
        signals,alltimes = signal,times
        snippits = []
        for signal,times in zip(signals,alltimes):
            N = len(signal)
            times = array(times)
            times = times[times>=window]
            times = times[times<N-window]
            t_o = -inf
            for t in times:
                if t-t_o>delta:
                    t_o = t
                    snippits.append(sig.zeromean(signal[t-window:t+window+1]))
    else: assert 0
    if biased:
        if multitaper: return fftppc_biased_multitaper(snippits,Fs,k),snippits
        else:          return fftppc_biased(snippits,Fs,taper=taper),snippits
    else:
        if multitaper: return fftppc_multitaper(snippits,Fs,k),snippits
        else:          return fftppc(snippits,Fs,taper=taper),snippits
    assert 0

def estimate_bias_in_uncorrected_ppc(
    signal,times,window=50,Fs=1000,nrand=100):
    '''
    Parameters
    ----------
    signal:
    times:
    
    Other Parameters
    ----------------
    window: positive int; deafult 50
    Fs: positive int; default 1000
    nrand: positive int; default 100
    
    Returns
    -------
    ff:
    bias:
    '''
    tried = []
    for i in range(nrand):
        ff,ppc = uncorrectedppc(phase_randomize(signal),times,window,Fs)
        tried.append(ppc)
    bias = mean(tried,0)
    return ff,bias

def phase_randomized_bias_correction(signal,times,window=50,Fs=1000,nrand=100):
    '''
    Estimates degrees of freedom using phase randomization.
    experimental.

    Parameters
    ----------
    signal:
    times:
    
    Other Parameters
    ----------------
    window: positive int; deafult 50
    Fs: positive int; default 1000
    nrand: positive int; default 100
    
    Returns
    -------
    ff:
    unbiased:
    '''
    warn('AS FAR AS WE KNOW THIS DOESNT REALLY WORK')
    ff,bias = estimate_bias_in_uncorrected_ppc(signal,times,window,Fs,nrand)
    K = 1.0-bias
    M = K/(1-K)
    print('estimated degrees of freedom:',M)
    print('nominal degrees of freedom=',len(times))
    ff,raw = uncorrectedppc(signal,times,window,Fs)
    unbiased = (raw*M-1)/(M-1)
    return ff, unbiased


def _temp_code_for_exploring_chance_level_delete_later():
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    # bias/ariance analysis
    # my understanding is that No. of samples should not change the mean PPC
    # value, but it will affect the variance. Since the PPC is computed as
    # a sum of complex unit vectors, we can reason about it's bias/variance
    # as a function of samples from a few simple simulations. This will replace
    # brute force simulations of chance level based on phase randomization.
    # The effect of multitaper on reducing the bias or variance is interesting.
    # I don't know how to account for that.
    nTapers = 1
    for nSamples in range(2,1000):
        simulated = []
        for i in range(100000):
            unbiased = (mean([abs(mean(exp(2*pi*1j*rand(nSamples)))) for t in range(nTapers)])**2*nSamples-1)/(nSamples-1)
            simulated.append(unbiased)
        simulated = sorted(simulated)
        print('n=%d 90%%=%f'%(nSamples,simulated[int(.90*len(simulated))]))
        print('n=%d 95%%=%f'%(nSamples,simulated[int(.95*len(simulated))]))
        print('n=%d 99%%=%f'%(nSamples,simulated[int(.99*len(simulated))]))
        print('n=%d 99.9%%=%f'%(nSamples,simulated[int(.999*len(simulated))]))
        clf()
        hist(simulated,100)
        draw()
        raw_input()

def ppc_chance_level(nSamples,nrandom,p,nTapers=1):
    '''
    **Caution:** This underestimates chance level if 
    spikes or the LFP signal are correlated in time.
    
    Parameters
    ----------
    nSamples: 
    nrandom:
    p: 
    
    Other Parameters
    ----------------
    nTapers: positive int; default 1
    
    Returns
    -------
    samples:
    '''
    raise DeprecationWarning(
        "`ppc_chance_level` underestimates the PPC chance "
        "level in the presence of spike-train or LFP "
        "autocorrelations and has been deprecated.")
    simulated = []
    for i in range(nrandom):
        unbiased = (
            mean([abs(mean(exp(2j*pi*rand(nSamples)))) 
            for t in range(nTapers)])**2*nSamples-1
            )/(nSamples-1)
        simulated.append(unbiased)
    return sorted(simulated)[int(p*len(simulated))]


def ppc_phase_randomize_chance_level_sample(
    signal,times,
    window=50,
    Fs=1000,
    k=4,multitaper=True,
    biased=False,
    delta=100,
    taper=None):
    '''
    Uses phase randomization to sample from the null hypothesis distribution.
    Returns the actual PPC samples rather than any summary statistics.
    You can do what you want with the distribution returned.

    Parameters
    ----------
    signal: 
        1D real valued signal
    times:  
        Times of events relative to signal
    window: 
        Time around event to examine
    
    Other Parameters
    ----------------
    window: positive int; default 50
        Time around event to examine
    Fs: positive int; default 1000
        sample rate for computing freqs
    k: positive int; default 4
        number of tapers
        Also accepts lists of signals / times
        returns (freqs, ppc, phase), lfp_segments
    multitaper: boolean; default True
    biased: booleanl default False
    delta: positive int; default 100
    taper: array; default None
    
    Returns
    -------
    freqs: npp.array
        Frequencies at which the PPC has been evaluated
        (phase-randomized samples)
    raw: np.array
        Raw (biased) value for the PPC at each frequency
        (phase-randomized samples)
    phases: np.array
        Phase values associated with each ppc coefficient
        (phase-randomized samples)
    '''
    if multitaper:
        print("Warning: multitaper can introduce a bias into PPC that depends on the number of tapers!")
        print("For a fixed number of tapers, the bias is constant, but be careful")

    if not taper is None and multitaper:
        print("A windowing taper was specified, but multitaper mode was also selected.")
        print("The taper argument is for providing a windowing function when not using multitaper estimation.")
        assert 0
    if type(taper) is types.FunctionType:
        taper = taper(window*2+1)
    if biased: warn('skipping bias correction entirely')
    assert window>0

    # need to compute PPC from phase randomized samples
    # We can't phase randomize the snippits because there may be some
    # correlation between them, we have to randomize the underlying LFP

    if len(shape(signal))==1:
        # only a single trial provided
        usetimes = discard_spikes_closer_than_delta(signal,times,delta)
        signal = phase_randomize(signal)
        snippits = array([sig.zeromean(signal[t-window:t+window+1]) for t in usetimes])
    elif len(shape(signal))==2:
        warn('assuming first dimension is trials / repititions')
        signals,alltimes = signal,times
        snippits = []
        for signal,times in zip(signals,alltimes):
            signal = phase_randomize(signal)
            N = len(signal)
            times = array(times)
            times = times[times>=window]
            times = times[times<N-window]
            t_o = -inf
            for t in times:
                if t-t_o>delta:
                    t_o = t
                    snippits.append(sig.zeromean(signal[t-window:t+window+1]))
    else: assert 0

    if biased:
        if multitaper: return fftppc_biased_multitaper(snippits,Fs,k),snippits
        else:          return fftppc_biased(snippits,Fs,taper=taper),snippits
    else:
        if multitaper: return fftppc_multitaper(snippits,Fs,k),snippits
        else:          return fftppc(snippits,Fs,taper=taper),snippits
    assert 0
