#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Routines for signal processing.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

from . import coherence
from . import conv
from . import morlet
from . import multitaper
from . import savitskygolay
from . import sde

import warnings
import numpy as np
from scipy.signal.signaltools import fftconvolve,hilbert
from scipy.signal import butter, filtfilt, lfilter
from scipy.interpolate import interp1d
from neurotools.util.array import find

import neurotools.util.array as narray # make_rebroadcast_slice

# Inverse of standard normal cumulative distribution function
from scipy.special import ndtri
from scipy.stats import rankdata
import scipy.stats
import scipy.interpolate as ip





############################################################
# Functions for Gaussian smoothing

def gaussian_kernel(sigma):
    '''
    Generate 1D Guassian kernel for smoothing
    
    Parameters
    ----------
    sigma : positive float
        Standard deviation of kernel. Kernel size is 
        automatically adjusted to ceil(sigma*2)*2+1 

    Returns
    -------
    K : vector
        normalized Gaussian kernel 
    '''
    assert sigma>0
    K = np.ceil(sigma*2)
    N = K*2+1
    K = np.exp( - (np.arange(N)-K)**2 / (2*sigma**2) )
    K *= 1./np.sum(K)
    return K

def gaussian_smooth(x,sigma,mode='same'):
    '''
    Smooth signal x with gaussian of standard deviation 
    sigma, using edge-clamped boundary conditions.
    
    Parameters
    ----------
    sigma: positive float
        Standard deviation of Gaussian smoothing kernel.
    x: 1D np.array
        Signal to filter. 

    Returns
    -------
    smoothed signal
    '''
    #x = concatenate([x::-1],x,x[::-1])
    K = gaussian_kernel(sigma)
    n = len(K)
    x = np.concatenate([np.ones(n)*x[0],x,np.ones(n)*x[-1]])
    #if len(x.shape)==1:
    x = np.convolve(x,K,mode=mode)
    #axes = tuple(sorted(list(set(range(len(x.shape)))-axis)))
    return x[n:-n]

def circular_gaussian_smooth(x,sigma):
    '''
    Smooth signal `x` with Gaussian of standard deviation 
    `sigma`,
    Circularly wrapped using Fourier transform.
    
    Parameters
    ----------
    x: np.array
        1D array-like signal
    sigma: positive float
        Standard deviation

    Returns
    -------
    smoothed signal
    '''
    N = len(x)
    g = np.exp(-np.linspace(-N/2,N/2,N)**2/sigma**2)
    g/= np.sum(g)
    f = np.fft.fft(g)
    return np.fft.fftshift(np.fft.ifft(np.fft.fft(x)*f).real)


def circular_gaussian_smooth_2D(x,sigma):
    '''
    Smooth signal x with gaussian of standard deviation sigma
    Circularly wrapped using Fourier transform
    
    sigma: standard deviation
    x: 2D array-like signal
    
    Parameters
    ----------
    x: np.ndarray
        Smoothing is performed over the last two dimensions, 
        which should have the same length
    sigma: positive float
        Standard deviation of Gaussian kernel for smoothing,
        in pixels. 
        
    Returns
    -------
    :np.array
        `x`, circularly smoothed over the last two 
        dimensions.
    '''
    # Make coordinate grid
    Nr,Nc = x.shape[-2:]
    gridr = np.linspace(-Nr/2,Nr/2,Nr)
    gridc = np.linspace(-Nc/2,Nc/2,Nc)
    
    # Make kernel
    dist  = abs(gridr[:,None]+gridc[None,:]*1j)**2
    g = np.exp(-dist/sigma**2)
    g/= np.sum(g)
    f = np.fft.fft2(np.fft.fftshift(g))
    
    # convolution via 2D FFT
    return np.fft.ifft2(np.fft.fft2(x)*f).real




















############################################################
# Functions for adjusting/scaling timeseries

def unitscale(signal,axis=None):
    '''
    Rescales `signal` so that its minimum is 0 and its 
    maximum is 1.

    Parameters
    ----------
    signal: np.array
        Array-like real-valued signal
    
    Returns
    -------
    signalL np.array
        Rescaled 
        `signal-min(signal)/(max(signal)-min(signal))`
    '''
    signal = np.float64(np.array(signal))
    if axis==None:
        # Old behavior
        signal-= np.nanmin(signal)
        signal/= np.nanmax(signal)
        return signal
    # New behavior
    theslice = narray.make_rebroadcast_slice(signal, axis)
    signal-= np.nanmin(signal,axis=axis)[theslice]
    signal/= np.nanmax(signal,axis=axis)[theslice]
    return signal

def topercentiles(x):
    '''

    Parameters
    ----------
    Returns
    -------
    '''
    n = len(x)
    x = np.array(x)
    order = x.argsort()
    ranks = order.argsort()
    return ranks/(len(x)-1)*100















############################################################
# Phase routines

def amp(x):
    '''
    Extracts amplitude envelope using Hilbert transform.
    `x` must be narrow-band. 
    No padding is performed --- watch out for boundary 
    effects.
    
    Parameters
    ----------
    x: np.array
        numeric time-series data
        
    Returns
    -------
    result:
        `abs(hilbert(x))`
    '''
    return np.abs(hilbert(np.array(x)))
    
def ifreq(x,Fs=1000,mode='pad'):
    '''
    Extract the instantaneous frequency from a narrow-band 
    signal using the Hilbert transform.
    
    Parameters
    ----------
    x: np.array
        numeric time-series data
    
    Other Parameters
    ----------------
    Fs : int
        defaults to 1000
    mode : str
        'pad' will return a signal of the original length
        'valid' will return a signal 1 sample shorter, with 
        derivative computed between each pair of points 
        in the original signal.
    '''
    pg = pghilbert(x)
    pg = pg/(2*np.pi)*Fs
    if mode=='valid':
        return pg # in Hz
    if mode=='pad':
        return fix_derivative(pg)
    assert 0
    
def pdiff(x):
    '''
    Take the derivative of a sequence of phases.
    Times when this derivative wraps around form 0 to 2π
    are correctly handeled.
    
    Parameters
    ----------
    x: np.array
        Array of phase values in radians to differentiate.

    Returns
    -------
    dx: np.array
        Phase derivative of `x`, with wrapping around 2π
        handled automatically. 
    '''
    return rewrap(np.diff(np.array(x)))

def rewrap(x):
    '''
    Used to handle wraparound when getting phase derivatives.
    See pdiff.
    
    Parameters
    ----------
    dx: np.array
        Array of phase derivatives, with 2π wrap-around
        not yet handled.
        
    Returns
    -------
    dx: np.array
        Phase derivative of `x`, with wrapping around 2π
        handled automatically. 
    '''
    x = np.array(x)
    return (x+np.pi)%(2*np.pi)-np.pi

def pghilbert(x):
    '''
    Extract phase derivative in time from a narrow-band 
    real-valued signal using the hilbert transform. 
    
    Parameters
    ----------
    x: np.float32
        Narrow-band real-valued signal to get the phase 
        gradient of. 
    
    Returns
    -------
    d/dt[phase(x)]: np.array
        Time derivative in radians/sample
    '''
    return pdiff(np.angle(np.hilbert(np.array(x))))

def unwrap(h):
    '''
    Unwraps a sequences of phase measurements so that,
    rather than ranging from 0 to 2π, 
    the values increase (or decrease) continuously.
    
    Parameters
    ----------
    h: np.float32
    
    Returns
    -------
    :np.array
        Re-wrapped phases, in radians. These phases
        will continue counting up/down and are not
        wrapped to [0,2π).
    '''
    return np.cumsum(fix_derivative(pdiff(h)))

def ang(x):
    '''
    Uses the Hilbert transform to extract the phase of x,
    in radians.
    X should be narrow-band. 
    
    Parameters
    ----------
    x: np.float32
        Narrow-band real-valued signal to get the phase 
        gradient of. 
    
    Returns
    -------
    ang: np.float32
        Phase angle of `x`, in radiance.
    '''
    return np.angle(np.hilbert(x))

def fix_derivative(x):
    '''
    Adjust discrete derivative `x` to pad-back the 1 sample
    removed by differentiation, and center the derivative
    on samples (rather than between them). 
    
    Applying this after using `diff()` is equivalent
    to using the differentiation kernel `[-.5,0,.5]`
    in the interior of the array and `[-1,1]` at its
    endpoints
    
    Parameters
    ----------
    x: np.array
        Derivative signal to fix.
    
    Returns
    -------
    :np.array
    '''
    x = np.array(x)
    n = len(x)+1
    result = np.zeros(n,dtype=x.dtype)
    result[1:]   += x
    result[:-1]  += x
    result[1:-1] *= 0.5
    return result

def phase_rotate(s,f,Fs=1000.):
    '''
    Only the phase advancement portion of a resonator.
    
    Parameters
    ----------
    s: np.array
        Analytic signal
    f: float
        Frequency, in Hz.
    
    Other Parameters
    ----------------
    Fs: positive number; default 1000
        Sampling rate.
    
    Returns
    -------
    :np.array
    '''
    theta = f*2*np.pi/Fs
    s *= np.exp(1j*theta)
    return s

def randband(N,fa=None,fb=None,Fs=1000):
    '''
    Returns Gaussian random noise band-pass filtered between 
    `fa` and `fb`.
    
    Parameters
    ----------
    N: int
        Number of samples to draw
    
    Returns
    -------
    '''
    return zscore(
            bandfilter(
            np.random.randn(N*2),fa=fa,fb=fb,Fs=Fs)
        )[N//2:N//2+N]

def phase_randomize(signal):
    '''
    Phase randomizes a signal by rotating frequency components by a random
    angle. Negative frequencies are rotated in the opposite direction.
    The nyquist frequency, if present, has it's sign randomly flipped.
    
    Parameters
    ----------
    signal: 1D np.array
    
    Returns
    -------
    '''
    assert 1==len(signal.shape)
    N = len(signal)
    '''
    if N%2==1:
        # signal length is odd.
        # ft will have one DC component then symmetric frequency components
        randomize  = np.exp(1j*np.random.rand((N-1)//2))
        conjugates = np.conj(randomize)[::-1]
        randomize  = np.append(randomize,conjugates)
    else:
        # signal length is even
        # will have one single value at the nyquist frequency
        # which will be real and can be sign flipped but not rotated
        flip = 1 if np.random.rand(1)<0.5 else -1
        randomize  = np.exp(1j*np.random.rand((N-2)//2))
        conjugates = np.conj(randomize)[::-1]
        randomize  = np.append(randomize,flip)
        randomize  = np.append(randomize,conjugates)
    # the DC component is not randomized
    randomize = np.append(1,randomize)
    # take FFT and apply phase randomization
    ff = np.fft.fft(signal)*randomize
    # take inverse
    randomized = np.fft.ifft(ff)
    return np.real(randomized)
    '''
    return phase_randomize_from_amplitudes(np.abs(np.fft.fft(signal)))

def phase_randomize_from_amplitudes(amplitudes):
    '''
    phase_randomize_from_amplitudes(amplitudes)
    treats input amplitudes as amplitudes of fourier components
    
    Parameters
    ----------
    Returns
    -------
    '''
    N = len(amplitudes)
    x = np.complex128(amplitudes) # need to make a copy
    if N%2==0: # N is even
        rephase = np.exp(1j*2*np.pi*np.random.rand((N-2)//2))
        rephase = np.concatenate([rephase,[np.sign(np.random.rand()-0.5)],np.conj(rephase[::-1])])
    else: # N is odd
        rephase = np.exp(1j*2*np.pi*np.random.rand((N-1)//2))
        rephase = np.append(rephase,np.conj(rephase[::-1]))
    rephase = np.append([1],rephase)
    x *= rephase
    return np.real(np.fft.ifft(x))

def sign_preserving_amplitude_demodulate(analytic_signal,doplot=False):
    '''
    Extracts an amplitude-modulated component from an analytic signal,
    Correctly flipping the sign of the signal when it crosses zero,
    rather than returning a rectified result.

    Sign-changes are heuriddddstically detected basd on the following:
        - An abnormally large skip in phase between two time points,
          larger than np.pi/2, that is also a local extremum in phase velocity
        - local minima in the amplitude at low-voltage with high curvature
    
    Parameters
    ----------
    analytic_signal: np.arrau
    
    Other Parameters
    ----------------
    doplot: boolean; default False
        Whether to draw plot
    
    Returns
    -------
    demodulated
    '''

    analytic_signal = zscore(analytic_signal)

    phase      = np.angle(analytic_signal)
    amplitude  = np.abs(analytic_signal)

    phase_derivative     = fix_derivative(pdiff(phase))
    phase_curvature      = fix_derivative(np.diff(phase_derivative))
    amplitude_derivative = fix_derivative(np.diff(amplitude))
    amplitude_curvature  = fix_derivative(np.diff(amplitude_derivative))

    amplitude_candidates = find( (amplitude_curvature >= 0.05) & (amplitude < 0.6) )
    amplitude_exclude    = find( (amplitude_curvature <  0.01) | (amplitude > 0.8) )
    phase_candidates     = find( (phase_curvature     >= 0.05) & (phase_derivative < np.pi*0.5) )
    phase_exclude        = find( (phase_derivative > np.pi*0.9) )
    aminima,_ = local_minima(amplitude)
    pminima,_ = local_minima(phase_derivative)
    pmaxima,_ = local_maxima(phase_derivative)
    minima = \
        ((set(aminima)|set(amplitude_candidates)) - \
          set(amplitude_exclude)) & \
        ((set(pminima)|set(pminima-1)|set(pmaxima)|set(pmaxima-1)) -\
          set(phase_exclude))

    minima = np.array(list(minima))
    minima = minima[diff(list(minima))!=1]

    edges = np.zeros(np.shape(analytic_signal),dtype=np.int32)
    edges[list(minima)] = 1
    sign = np.cumsum(edges)%2*2-1

    demodulated = amplitude*sign

    if doplot:
        clf()

        Nplots = 4
        iplot = 1

        subplot(Nplots,1,iplot)
        iplot+=1
        plot(demodulated,color='r',lw=2)
        [axvline(x,lw=2,color='k') for x in (minima)]

        subplot(Nplots,1,iplot)
        iplot+=1
        plot(phase_derivative,color='r',lw=2)
        [axvline(x,lw=2,color='k') for x in (minima)]

        subplot(Nplots,1,iplot)
        iplot+=1
        plot(amplitude_curvature,color='r',lw=2)
        [axvline(x,lw=2,color='k') for x in (minima)]

        subplot(Nplots,1,iplot)
        iplot+=1
        plot(real(analytic_signal),color='g')
        [axvline(x,lw=2,color='k') for x in (minima)]

    return demodulated








    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
############################################################
# Filtering

def nonnegative_bandpass_filter(data,fa=None,fb=None,
    Fs=1000.,order=4,zerophase=True,bandstop=False,
    offset=1.0):
    '''
    For filtering data that must remain non-negative. Due to 
    ringing conventional fitering can create values less 
    than zero for non-negative real inputs. 
    This may be unrealistic for some data.

    To compensate, this performs the filtering on the 
    natural logarithm of the input data. For small numbers, 
    this can lead to numeric underflow, so an offset 
    parameter (default 1) is added to the data for 
    stability.

    Parameters
    ----------
    data (ndarray): 
        data, filtering performed over last dimension
    fa (number): 
        low-freq cutoff Hz. If none, lowpass at fb
    fb (number): 
        high-freq cutoff Hz. If none, highpass at fa
    Fs (int): 
        Sample rate in Hz
        
    Other Parameters
    ----------------
    order (1..6): 
        butterworth filter order. Default is 4
    zerophase (boolean): 
        Use forward-backward filtering? (true)
    bandstop (boolean): 
        Do band-stop rather than band-pass
    offset (positive number): 
        Offset data to avoid underflow (1)
    
    Returns
    -------
    filtered : 
        Filtered signal
    '''
    offset -= 1.0
    data = np.log1p(data+offset)
    filtered = bandpass_filter(data,
        fa=fa, fb=fb, Fs=Fs,
        order=order,
        zerophase=zerophase,
        bandstop=bandstop)
    return np.expm1(filtered)


def bandpass_filter(data,fa=None,fb=None,
    Fs=1000.,order=4,zerophase=True,bandstop=False):
    '''
    IF fa is None, assumes lowpass with cutoff fb
    IF fb is None, assume highpass with cutoff fa
    Array can be any dimension, filtering performed over 
    last dimension.

    Args:
        data (ndarray): data, filtering performed over last dimension
        fa (number): low-frequency cutoff. If none, highpass at fb
        fb (number): high-frequency cutoff. If none, lowpass at fa
        order (1..6): butterworth filter order. Default 4
        zerophase (boolean): Use forward-backward filtering? (true)
        bandstop (boolean): Do band-stop rather than band-pass
    
    Returns
    -------
    result: np.array
        Filtered signal. 
    '''
    if np.product(data.shape)<=0:
        raise ValueError('Singular array! no data to filter')
    N = data.shape[-1]
    if N<=1:
        raise ValueError('Filters over last dimension, which should have len>1')
    padded = np.zeros(data.shape[:-1]+(2*N,),dtype=data.dtype)
    padded[...,N//2  :N//2+N] = data
    padded[...,     :N//2  ] = data[...,N//2:0    :-1]
    padded[...,N//2+N:     ] = data[...,-1 :N//2-1:-1]
    if not fa is None and not fb is None:
        if bandstop:
            b,a = butter(order,np.array([fa,fb])/(0.5*Fs),btype='bandstop')
        else:
            b,a = butter(order,np.array([fa,fb])/(0.5*Fs),btype='bandpass')
    elif not fa==None:
        # high pass
        b,a  = butter(order,fa/(0.5*Fs),btype='high')
        assert not bandstop
    elif not fb==None:
        # low pass
        x = fb/(0.5*Fs)-1e-10
        if x>=1:
            raise ValueError('The low-frequency cutoff is larger than the nyquist freqency?')
        b,a  = butter(order,x,btype='low')
        assert not bandstop
    else: raise Exception('Both fa and fb appear to be None')
    method = filtfilt if zerophase else lfilter
    # hide the scipy/numpy compatibility future warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        result = method(b,a,padded)
        return result[...,N//2:N//2+N]
    assert 0


def box_filter(data,smoothat,padmode='reflect'):
    '''
    Smooths data by convolving with a size smoothat box
    provide smoothat in units of frames i.e. samples. 
    
    Parameters
    ----------
    x : np.array
        One-dimensional numpy array of the signal to be filtred
    window : positive int
        Filtering window length in samples
    mode : string, default 'same'
        If 'same', the returned signal will have the same time-base and
        length as the original signal. if 'valid', edges which do not
        have the full window length will be trimmed
    
    Returns
    -------
    np.array :
        One-dimensional filtered signal
    '''
    data = np.array(data)
    data = np.float32(data)
    N = len(data)
    assert len(data.shape)==1
    padded = np.zeros(2*N,dtype=data.dtype)
    padded[N//2:N//2+N]=data
    if padmode=='reflect':
        padded[:N//2]=data[N//2:0:-1]
        padded[N//2+N:]=data[-1:N//2-1:-1]
    else:
        padded[:N//2]=data[0]
        padded[N//2+N:]=data[-1]
    smoothed = fftconvolve(padded,np.ones(smoothat)/float(smoothat),'same')
    return smoothed[N//2:N//2+N]

def median_filter(x,window=100,mode='same'):
    '''
    median_filter(x,window=100,mode='same')
    Filters a signal by calculating the median in a sliding window of
    width 'window'

    mode='same' will compute median even at the edges, where a full window
        is not available

    mode='valid' will compute median only at points where the full window
        is available

    Parameters
    ----------
    x : np.array
        One-dimensional numpy array of the signal to be filtred
    window : positive int
        Filtering window length in samples
    mode : string, default 'same'
        If 'same', the returned signal will have the same time-base and
        length as the original signal. if 'valid', edges which do not
        have the full window length will be trimmed
    
    Returns
    -------
    np.array :
        One-dimensional filtered signal
    '''
    x = np.array(x)
    n = x.shape[0]
    if mode=='valid':
        filtered = [np.median(x[i:i+window]) for i in range(n-window)]
        return np.array(filtered)
    if mode=='same':
        a = window // 2
        b = window - a
        filtered = [np.median(x[max(0,i-a):min(n,i+b)]) for i in range(n)]
        return np.array(filtered)
    assert 0

def percentile_filter(x,pct,window=100,mode='same'):
    '''
    percentile_filter(x,pct,window=100,mode='same')
    Filters a signal by calculating the median in a sliding window of
    width 'window'

    mode='same' will compute median even at the edges, where a full window
        is not available

    mode='valid' will compute median only at points where the full window
        is available

    Parameters
    ----------
    x : np.array
        One-dimensional numpy array of the signal to be filtred
    pct: float in 0..100
        Percentile to apply
    window : positive int
        Filtering window length in samples
    mode : string, default 'same'
        If 'same', the returned signal will have the same time-base and
        length as the original signal. if 'valid', edges which do not
        have the full window length will be trimmed
    
    Returns
    -------
    np.array :
        One-dimensional filtered signal
    '''
    x = np.array(x)
    n = x.shape[0]
    if mode=='valid':
        filtered = [np.percentile(x[i:i+window],pct) for i in range(n-window)]
        return np.array(filtered)
    if mode=='same':
        a = window // 2
        b = window - a
        filtered = [np.percentile(x[max(0,i-a):min(n,i+b)],pct) for i in range(n)]
        return np.array(filtered)
    assert 0

def variance_filter(x,window=100,mode='same'):
    '''
    Extracts signal variance in a sliding window

    mode='same' will compute median even at the edges, where a full window
    is not available

    mode='valid' will compute median only at points where the full window
    is available
    
    Parameters
    ----------
    x : np.array
        One-dimensional numpy array of the signal to be filtred
    window : positive int
        Filtering window length in samples
    mode : string, default 'same'
        If 'same', the returned signal will have the same time-base and
        length as the original signal. if 'valid', edges which do not
        have the full window length will be trimmed
    
    Returns
    -------
    np.array :
        One-dimensional filtered signal
    '''
    x = np.array(x)
    n = x.shape[0]
    if mode=='valid':
        filtered = [np.var(x[i:i+window]) for i in range(n-window)]
        return array(filtered)
    if mode=='same':
        a = window // 2
        b = window - a
        filtered = [np.var(x[max(0,i-a):min(n,i+b)]) for i in range(n)]
        return np.array(filtered)
    assert 0


def stats_block(data,statfunction,N=100,sample_match=None):
    '''
    Compute function of signal in blocks of size $N$ over the last axis
    of the data
    
    Parameters
    ----------
    data: np.array
        N-dimensional numpy array. Blocking is performed over the last axis
    statfunction: function
        Statistical function to compute on each block. Should be, or 
        behave similarly to, the `numpy` buit-ins, e.g. np.mean,
        np.median, etc.
    
    Other Parameters
    ----------------
    N: positive integer, default 100
        Block size in which to break data. If data cannot be split 
        evenly into blocks of size $N$, then data are truncated to the 
        largest integer multiple of N. 
    sample_match: positive integer, default None
        If not None, then blocks will be sub-sampled to contain
        `sample_match` samples. `sample_match` should not exceed
        data.shape[-1]//N
    
    Returns
    -------
    np.array : 
        Blocked data
    '''
    N = int(N)
    L = data.shape[-1]
    B = L//N
    if not sample_match is None and sample_match>N:
        raise ValueError('%d samples/block requested but blocks have only %d samples'%(sample_match,N))
    D = B*N
    data = data[...,:D]
    data = np.reshape(data,data.shape[:-1]+(B,N))
    if not sample_match is None:
        keep = np.int32(np.linspace(0,N-1,sample_match))
        data = data[:,keep]
    return statfunction(data,axis=-1)

def mean_block(data,N=100,sample_match=None):
    '''
    Calls stats_block using np.mean. See documentation of stats_block for
    details.
    
    Parameters
    ----------
    data: 1D np.array
        Signal to filter
    N: positive integer, default 100
        Block size in which to break data. If data cannot be 
        split evenly into blocks of size $N$, then data are 
        truncated to the largest integer multiple of N. 
    sample_match: positive integer, default None
        If not None, then blocks will be sub-sampled to 
        contain `sample_match` samples. `sample_match` 
        should not exceed data.shape[-1]//N
        
    Returns
    -------
    result: np.array
        `stats_block(data,np.mean,N)`
    '''
    return stats_block(data,np.mean,N,sample_match)

def var_block(data,N=100):
    '''
    Calls stats_block using `np.var`. 
    See documentation of `stats_block` for details.
    
    Parameters
    ----------
    data: 1D np.array
        Signal to filter
    N: positive integer, default 100
        Block size in which to break data. If data cannot be split 
        evenly into blocks of size $N$, then data are truncated to the 
        largest integer multiple of N. 
    sample_match: positive integer, default None
        If not None, then blocks will be sub-sampled to contain
        `sample_match` samples. `sample_match` should not exceed
        data.shape[-1]//N
        
    Returns
    -------
    result: np.array
        `stats_block(data,np.var,N)`
    '''
    return stats_block(data,np.var,N)

def median_block(data,N=100):
    '''
    Calls stats_block using np.median. See documentation of stats_block for
    details.
    
    Parameters
    ----------
    data: 1D np.array
        Signal to filter
    N: positive integer, default 100
        Block size in which to break data. If data cannot be split 
        evenly into blocks of size $N$, then data are truncated to the 
        largest integer multiple of N. 
        
    Returns
    -------
    result: np.array
        `stats_block(data,np.median,N)`
    '''
    return stats_block(data,np.median,N)

def linfilter(A,C,x,initial=None):
    '''
    Linear response filter on data $x$ for system
    
    $$
    \\partial_t z = A z + C x(t)
    $$
    
    Parameters
    ----------
    A : matrix
        K x K matrix defining linear syste,
    C : matrix
        K x N matrix defining projection from signal $x$ to linear system
    x : vector or matrix
        T x N sequence of states to filter
    initial : vector
        Optional length N vector of initial filter conditions. Set to 0
        by default

    Returns
    -------
    filtered : array
        filtered data
    '''
    # initial state for filters (no response)
    L = len(x)
    K = A.shape[0]
    z = np.zeros((K,1)) if initial is None else initial
    filtered = []
    for t in range(L):
        dz = A.dot(z) + C.dot([[x[t]]])
        z += dz
        filtered.append(z.copy())
    return np.squeeze(np.array(filtered))
    
def padout(data):
    '''
    Generates a reflected version of a 1-dimensional signal. 

    The original data is placed in the middle, between the 
    mirrord copies.
    Use the function "padin" to strip the padding
    
    Parameters
    ----------
    data: 1d np.array
    
    Returns
    -------
    result: np.array
        length 2*data.shape[0] padded array
    '''
    N = len(data)
    assert len(data.shape)==1
    padded = np.zeros(2*N,dtype=data.dtype)
    padded[N//2  :N//2+N]=data
    padded[     :N//2  ]=data[N//2:0    :-1]
    padded[N//2+N:     ]=data[-1 :N//2-1:-1]
    return padded

def padin(data):
    '''
    Removes padding added by the `padout` function; 
    `padin` and `padout` together are used to control the
    boundary condtitions for filtering. See the documentation
    for padout for details.
    
    Parameters
    ----------
    data : array-like
        Data array produced by the `padout function` 
        
    Returns
    -------
    np.array : 
        data with edge padding removed
    '''
    N = len(data)
    assert len(data.shape)==1
    return data[N//2:N//2+N]

def estimate_padding(fa,fb,Fs=1000):
    '''
    Estimate the amount of padding needed to address boundary conditions
    when filtering. Takes into account the filter bandwidth, which is
    related to the time-locality of the filter, and therefore the amount
    of padding needed to prevent artifacts at the edge.
    
    Parameters
    ----------
    Returns
    -------
    '''
    bandwidth  = fb if fa is None else fa if fb is None else min(fa,fb)
    wavelength = Fs/bandwidth
    padding    = int(np.ceil(2.5*wavelength))
    return padding

def lowpass_filter(x, cut=10, Fs=1000, order=4):
    '''
    Execute a butterworth low pass Filter at frequency "cut"
    Defaults to order=4 and Fs=1000
    
    Parameters
    ----------
    Returns
    -------
    '''
    return bandpass_filter(x,fb=cut,Fs=Fs,order=order)

def highpass_filter(x, cut=40, Fs=1000, order=4):
    '''
    Execute a butterworth high pass Filter at frequency "cut"
    Defaults to order=4 and Fs=1000
    
    Parameters
    ----------
    Returns
    -------
    '''
    return bandpass_filter(x,fa=cut,Fs=Fs,order=order)

def fdiff(x,Fs=240.):
    '''
    Take the discrete derivative of a signal, correcting 
    result for sample rate. This procedure returns a singnal
    two samples shorter than the original.
    
    Parameters
    ----------
    x: 1D np.array
        Signal to differentiate
    Fs: positive number; 
        Sampling rate of x

    Returns
    -------
    '''
    return (x[2:]-x[:-2])*Fs*.5






















    







############################################################
# Binary signal functions

def arenear(b,K=5):
    '''
    Expand a boolean/binary sequence by K samples in each 
    direction.
    See also "aresafe"
    
    Parameters
    ----------
    b: 1D np.bool
        Boolean array; 
        
    Other Parameters
    ----------------
    K: positive int; default 5
        Number of samples to add to each end of 
        spans of `b` which are `True`
    
    Returns
    -------
    b: bp.bool
        Expanded `b`.
    '''
    for i in range(1,K+1):
        b[i:] |= b[:-i]
    for i in range(1,K+1):
        b[:-i] |= b[i:]
    return b

def aresafe(b,K=5):
    '''
    Contract a boolean/binary sequence by `K` samples in 
    each direction. I.e. trim off `K` samples from the ends
    of spans of `b` that are `True`.
    
    For example, you may want to test for a condition, but 
    avoid samples close to edges in that condition.
    
    Parameters
    ----------
    b: 1D np.bool
        Boolean array; 
        
    Other Parameters
    ----------------
    K: positive int; default 5
        Number of samples to shave off each end of 
        spans of `b` which are `True`
        
    Returns
    -------
    b: bp.bool
        Trimmed `b`
    '''
    for i in range(1,K+1):
        b[i:] &= b[:-i]
    for i in range(1,K+1):
        b[:-i] &= b[i:]
    return b

def get_edges(signal,pad_edges=True):
    '''
    Assuming a binary signal, get the start and stop times 
    of each treatch of "1s"
    
    Parameters
    ----------
    signal : 1-dimensional array-like
    
    Other Parameters
    ----------------
    pad_edges : True
        Should we treat blocks that start or stop at the 
        beginning or end of the signal as valid?
    
    Returns
    -------
    2xN array of bin start and stop indecies
    '''
    if len(signal)<1:
        return np.array([[],[]])
    if tuple(sorted(np.unique(signal)))==(-2,-1):
        raise ValueError('signal should be bool or int∈{0,1}; (did you use ~ on an int array?)')
    signal = np.int32(np.bool8(signal))
    starts = list(np.where(np.diff(np.int32(signal))==1)[0]+1)
    stops  = list(np.where(np.diff(np.int32(signal))==-1)[0]+1)
    if pad_edges:
        # Add artificial start/stop time to incomplete blocks
        if signal[0 ]: starts = [0]   + starts
        if signal[-1]: stops  = stops + [len(signal)]
    else:
        # Remove incomplete blocks
        if signal[0 ]: stops  = stops[1:]
        if signal[-1]: starts = starts[:-1]
    return np.array([np.array(starts), np.array(stops)])

def set_edges(edges,N):
    '''
    Converts list of start, stop times over time period N into a [0,1]
    array which is 1 for any time between a [start,stop)
    edge info outsize [0,N] results in undefined behavior
    
    Parameters
    ----------
    Returns
    -------
    '''
    x = np.zeros(shape=(N,),dtype=np.int32)
    for (a,b) in edges:
        x[a:b]=1
    return x

def remove_gaps(w,cutoff):
    '''
    Removes gaps (streaches of zeros bordered by ones) from 
    binary signal `w` that are shorter than `cutoff` in duration.
    
    Parameters
    ----------
    w : one-dimensional array-like
        Binary signal
    cutoff : positive int
        Minimum gap duration to keep
        
    Returns
    -------
    array-like
        Copy of w with gaps shorter than `cutoff` removed
    '''
    a,b  = get_edges(1-w)
    gaps = b-a
    keep = np.array([a,b])[:,gaps>cutoff]
    newgaps = set_edges(keep.T,len(w))
    return 1-newgaps
    
def remove_short(w,cutoff):
    '''
    Removes spans of ones bordered by zeros from 
    binary signal `w` that are shorter than `cutoff` in duration.
    
    Parameters
    ----------
    w : one-dimensional array-like
        Binary signal
    cutoff : positive int
        Minimum gap duration to keep
        
    Returns
    -------
    array-like
        Copy of w with spans shorter than `cutoff` removed
    '''
    a,b  = get_edges(w)
    gaps = b-a
    keep = np.array([a,b])[:,gaps>cutoff]
    newgaps = set_edges(keep.T,len(w))
    return newgaps

def pieces(x,thr=4):
    '''
    Chops up `x` between points that differ by more
    than `thr`
    
    Parameters
    ----------
    x: 1D np.array
    
    Other Parameters
    ----------------
    thr: number; default 4
        Derivative threshold for cutting segments.
    
    Returns
    -------
    ps: list
     List of `(range(a,b),x[a:b])` for each piece of `x`.
    '''
    dd = diff(x)
    br = [0]+list(find(abs(dd)>thr))+[len(x)]
    ps = []
    for i in range(len(br)-1):
        a,b = br[i:][:2]
        a+=1
        if a==b: continue
        ps.append((range(a,b),x[a:b]))
    return ps
    






















############################################################
# Signal cleaning

def interpolate_NaN(u):
    '''
    Fill in NaN (missing) data in a one-dimensional 
    timeseries via linear interpolation.
    
    Parameters
    ----------
    u: np.array
        Signal in which to interpolate NaN values
    
    Returns
    -------
    u: np.array
        Copy of `u` with NaN values filled in via linear
        interpolation.
    '''
    u = np.array(u)
    for s,e in list(zip(*get_edges(~np.isfinite(u)))):
        if s==0: continue
        if e==len(u): continue
        a = u[s-1]
        b = u[e]
        u[s:e+1] = a + (b-a)*np.linspace(0,1,e-s+1)
    return u

def interpolate_NaN_quadratic(u):
    '''
    Fill in NaN (missing) data in a one-dimensional 
    timeseries via quadratic interpolation.
    
    Parameters
    ----------
    u: 1D np.array
    
    Returns
    -------
    :np.array
        `u`, with `NaN` values replaced using locally-
        quadratic interpolation.
    
    '''
    u = np.array(u)
    N = len(u)
    i = np.arange(N)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", 
            message="Polyfit may be poorly conditioned")
        splices = []
        for s,e in list(zip(*get_edges(~np.isfinite(u)))):
            #if s==0: continue
            #a = u[s-1]
            #b = u[e]
            length = e-s
            ist   = max(0,s-length//2-1)
            ind   = min(N-1,e+length//2)
            ix    = np.arange(ist,ind+1)
            iy    = u[ist:ind+1]
            ok    = np.isfinite(iy)
            a,b,c = np.polyfit(ix[ok],iy[ok],2);
            yhat  = a*ix**2+b*ix+c
            assert np.all(np.isfinite(yhat))
            splices.append((s,e+1,yhat[s-ist:e-ist+1]))
        for a,b,u2 in splices:
            u[a:b]=u2
        assert np.all(np.isfinite(u))
        return u

def killSpikes(x,threshold=1):
    '''
    Remove times when the signal exceeds a given threshold of the
    standard deviation of the underlying signal. Removed data are
    re-interpolated from the edges. This procedure is particularly
    useful in correcting higkinematics velocity trajectories. Velocity
    should be smooth, but motion tracking errors can cause sharp spikes
    in the signal.
    
    Parameters
    ----------
    x:
    threshold:1
    
    Returns
    -------
    '''
    x = np.array(x)
    y = zscore(highpass_filter(x))
    x[abs(y)>threshold] = nan
    for s,e in zip(*get_edges(isnan(x))):
        a = x[s-1]
        b = x[e+1]
        print(s,e,a,b)
        x[s:e+2] = np.linspace(a,b,e-s+2)
    return x


def drop_nonfinite(x):
    '''
    Flatten array and remove non-finite values

    Parameters
    ----------
    x: np.float32
        Numpy array from which to move non-finite values
        
    Returns
    1D np.float32
        Flattened array with non-finite values removed.
    '''
    x = np.float32(x).ravel()
    return x[np.isfinite(x)]



def virtual_reference_line_noise_removal(lfps,frequency=60,hbw=5):
    '''
    Accepts an array of LFP recordings (first dimension should be 
    channel number, second dimension time ).
    Sample rate assumed 1000Hz
    
    Extracts the mean signal within 2.5 Hz of 60Hz.
    For each channel, removes the projection of the LFP signal onto 
    this estimated line noise signal.
    
    I've found this approach sometimes doesn't work very well, 
    so please inspect the output for quality. 
    
    To filter out overtones, see `band_stop_line_noise_removal()`.
    
    Parameters
    ----------
    lfps:
        LFP channel data
    frequency: positive number
        Line noise frequency, defaults to 60 Hz
        (USA).  
    hbw: positive number
        Half-bandwidth settings; Default is 5
        
    Returns
    -------
    removed: np.array
        Band-stop filtered signal
    '''
    filtered = [bandfilter(x,frequency-hbw,frequency+hbw) for x in lfps]
    noise    = mean(filtered,0)
    scale    = 1./dot(noise,noise)
    removed  = [x-dot(x,noise)*scale*noise for x in lfps]
    return removed

def band_stop_line_noise_removal(lfps,frequency=60.):
    '''
    Remove line noise using band-stop at 60Hz 
    and overtones.
    
    Parameters
    ----------
    lfps:
        LFP channel data
    frequency: positive number
        Line noise frequency, defaults to 60 Hz
        (USA).  
    hbw: positive number
        Half-bandwidth settings; Default is 10
        
    Returns
    -------
    removed: np.array
        Band-stop filtered signal
    '''
    hbw   = 10
    freqs = float32([60,120,180,240,300])*frequency/60.
    lfps = array(lfps)
    for i,x in enumerate(lfps):
        for f in freqs:
            lfps[i,:] = bandfilter(lfps[i,:],f-hbw,f+hbw,bandstop=1)
    return lfps
    
    















############################################################
# minima/maxima

def local_peak_within(freqs,cc,fa,fb):
    '''
    For a spectrum, identify the largest local maximum in 
    the frequency range [fa,fb].
    
    Parameters
    ----------
    freqs : np.array
        Frequencies
    cc : np.array
        Amplitude
    fa : float
        low-frequency cutoff
    fb : float
        high-frequency cutoff
    
    Returns
    -------
    i: 
        index of peak, or None if no local peak found
    frequency:
        frequency at peak, or None if no local peak found
    peak:
        amplitude at peak, or None if no local peak found
    
    '''
    local = local_maxima(cc)[0]
    peaks = list(set(local) & set(find((freqs>=fa) & (freqs<=fb))))
    if len(peaks)==0: 
        return (None,)*3 # no peaks!
    i     = peaks[argmax(cc[peaks])]
    return i, freqs[i], cc[i]
    
def local_maxima(x):
    '''
    Detect local maxima in a 1D signal.

    Parameters
    ----------
    x: np.array
    
    Returns
    -------
    t: np.int32
        Location of local maxima in x
    x[t]: np.array
        Values of `x` at these local maxima.
    '''
    t = np.where(np.diff(np.sign(np.diff(x)))<0)[0]+1
    return t,x[t]

def local_minima(x):
    '''
    Detect local minima in a 1D signal.

    Parameters
    ----------
    x: np.array
    
    Returns
    -------
    t: np.int32
        Location of local minima in x
    x[t]: np.array
        Values of `x` at these local minima.
    '''
    t,x = local_maxima(-x)
    return t,-x

def peak_within(freqs,spectrum,fa,fb):
    '''
    Find maximum within a band
    
    Parameters
    ----------
    freqs : np.array
        Frequencies
    spectrum : 
    fa : float
        low-frequency cutoff
    fb : float
        high-frequency cutoff
    
    Returns
    -------
    '''
    # clean up arguments
    order    = argsort(freqs)
    freqs    = np.array(freqs)[order]
    spectrum = np.array(spectrum)[order]
    start = find(freqs>=fa)[0]
    stop  = find(freqs<=fb)[-1]+1
    index = np.argmax(spectrum[start:stop]) + start
    return freqs[index], spectrum[index]


def interpmax1d(x):
    '''
    Locate a peak in a 1D array by interpolation; see
    dspguru.com/dsp/howtos/how-to-interpolate-fft-peak
    
    Parameters
    ----------
    x: 1D np.array; Signal in which to locate the gloabal maximum.
    
    Returns
    -------
    i: float; Interpolated index of global maximum in `x`.
    '''
    i = np.argmax(x)
    try:
        y1,y2,y3 = x[i-1:i+2]
        return i + (y3-y1)/(2*(2*y2-y1-y3))
    except:
        return i

















############################################################
# 1D signal adjusting

def zeromean(x,axis=0,verbose=False,ignore_nan=True):
    '''
    Remove the mean trend from data
    
    Parameters
    ----------
    x : np.array
        Data to remove mean trend from
    
    Other Parameters
    ----------------
    axis : int or tuple, default None
        Axis over which to take the mean; 
        forwarded to np.mean axis parameter
    
    Returns
    -------
    '''
    x = np.array(x)
    if np.prod(x.shape)==0:
        return x
    theslice = narray.make_rebroadcast_slice(x,axis=axis,verbose=verbose)
    return x-(np.nanmean if ignore_nan else np.mean)(x,axis=axis)[theslice]

def zscore(x,axis=0,regularization=1e-30,verbose=False,ignore_nan=True):
    '''
    Z-scores data, defaults to the first axis.
    A regularization factor is added to the standard deviation to preven
    numerical instability when the standard deviation is extremely small.
    The default refularization is 1e-30.
    
    Parameters
    ----------
    x:
        Array-like real-valued signal.
    axis: 
        Axis to zscore; default is 0.

    Returns
    -------
    x: np.ndarray
        (x-mean(x))/std(x)
    '''
    x = zeromean(x,axis=axis,ignore_nan=ignore_nan)
    if np.prod(x.shape)==0:
        return x
    theslice = narray.make_rebroadcast_slice(x,axis=axis,verbose=verbose)
    ss = (np.nanstd if ignore_nan else np.std)(x,axis=axis)+regularization
    return x/ss[theslice]
    
def gaussianize(x,axis=-1,verbose=False):
    '''
    Use percentiles to force a timeseries to have a normal 
    distribution.
    '''
    x = np.array(x)
    if np.prod(x.shape)==0: return x
    return ndtri((rankdata(x,axis=axis))/(x.shape[axis]+1))

def uniformize(x,axis=-1,killeps=None):
    '''
    Use percentiles to force a timeseries to have a uniform 
    [0,1] distribution.
    
    `uniformize()` was designed to operate on non-negative 
    data and  has some quirks that have been retained for 
    archiving and backwards compatibility.
    
    Namely, if the `killeps` argument is provided,  
    `uniformize()` assumes that inputs are non-
    negative and excludes values less than `killeps`*σ,
    where σ is the standard-deviation of `x`, from the
    percentile rankings. The original default for
    `killeps` was `1e-3`.
    
    This was done because in the T-maze experiments, 
    the mouse often spends quite a bit of time at the
    beginning of the maze. We don't want most of the [0,1]
    dynamic range dedicatde to encoding positions or times
    near the start of trials. So, we estimate the scale
    of `x` using its standard deviation, and then clip
    values that are small relative to this scale. 
    This heuristic works for position/time data from the 
    Dan-Helen-Ethan experiments, but isn't very general.
    
    Parameters
    ----------
    x: np.float32:
        Timeseries
        
    Other Parameters
    ----------------
    axis: axis specifies; default -1
        axis argument forwarded to ` scipy.stats.rankdata`
    killeps: positive float; default None
        Original value of `killeps` passed to 
        `uniformize()`
    '''
    x_ = np.float32(x)
    if np.min(x_)<0.0:
        raise ValueError(
            '`uniformize` was written for non-negative '
            'values only and has some quirks that have been'
            ' retained for backwards compatibility.')
    if killeps is not None:
        x_[np.abs(x)<killeps*std(x)] = NaN
    ranks  = scipy.stats.rankdata(x_,axis=axis)
    
    # NaN end up ranked at the end (high ranks)
    # Detect them and set their rank to zero, since
    # these correspond to small clipped values.
    used = np.sum(np.isfinite(x_))
    ranks[ranks>used] = 0
    
    scaled = np.clip(ranks/used,0,1)
    return scaled

def invert_uniformize(x,p,axis=-1,killeps=None):
    '''
    Inverts the `uniformize()` function
    
    `uniformize()` was designed to operate on non-negative 
    data and has some quirks that have been retained for 
    archiving and backwards compatibility.
    
    Namely, if the `killeps` argument is provided,  
    `uniformize()` assumes that inputs are non-
    negative and excludes values less than `killeps`*σ,
    where σ is the standard-deviation of `x`, from the
    percentile rankings. The original default for
    `killeps` was `1e-3`.
    
    This was done because in the T-maze experiments, 
    the mouse often spends quite a bit of time at the
    beginning of the maze. We don't want most of the [0,1]
    dynamic range dedicate to encoding positions or times
    near the start of trials. So, we estimate the scale
    of `x` using its standard deviation, and then clip
    values that are small relative to this scale. 
    This heuristic works for position/time data from the 
    Dan-Helen-Ethan experiments, but isn't very general.
    
    Uniformize processing steps
    
     - Mark timepoints where abs(x)<killeps*std(x)
     - Rank data excluding these timepoints
     - Check how many timepoints were actually included
     - Normalize ranks to this amount
    
    Parameters
    ----------
    x: np.float32:
        Original timeseries passed as argument `x` to
        `uinformize()`
    p: np.float32 ∈ [0,1]
        Values on [0,1] interval to convert back into
        raw signal values, based on percentiles of 
        `x`. 
        
    Other Parameters
    ----------------
    axis: axis specifies; default -1
        axis argument forwarded to ` scipy.stats.rankdata`
    killeps: positive float; default None
        Original value of `killeps` passed to 
        `uniformize()` (leave blank if you did not specify
        this argument; It defaults to 1e-3)
        
    Returns
    -------
    np.float32
        Recontructed values. 
        This should be equivalent to the original data
        with values less than `killeps` times the 
        standard-deviation "σ" of `original_x` set to
        σ*killeps
    '''
    x_ = np.float32(x)
    if np.min(x_)<0.0:
        raise ValueError(
            '`uniformize` was written for non-negative '
            'values only and has some quirks that have been'
             'retained for backwards compatibility.')
    p = np.float32(p)
    if np.any(p<0.0) or np.any(p>1.0):
        raise ValueError(
            '`p` Should contain values in [0,1], but ranges'
            ' over %s'%(np.min(p),np.max(p)))
        
    if killeps is not None:
        x_[np.abs(x)<killeps*std(x)] = NaN
    ranks = scipy.stats.rankdata(x_,axis=axis)
    
    # NaN end up ranked at the end (high ranks)
    # Detect them and set their rank to zero, since
    # these correspond to small clipped values.
    used = np.sum(np.isfinite(x_))
    ranks[ranks>used] = 0
    
    scaled = np.clip(ranks/used,0,1)

    return ip.interp1d(
        scaled,x_,bounds_error=False,fill_value=(0,1)
    )(p)


def deltaovermean(x,axis=0,
    regularization=1e-30,verbose=False,ignore_nan=True):
    '''
    Subtracts, then divides by, the mean.
    Sometimes called "dF/F"
    
    Parameters
    ----------
    x: np.array
        Array-like real-valued signal.
        
    Other Parameters
    ----------------
    axis: maptlotlib.Axis
        Axis to zscore; default is 0.
    regularization: positive number; default 1e-30
        Regularization to avoid division by 0
    verbose: boolean; default False
    ignore_nan: boolean; default True
    
    Returns
    -------
    x: np.ndarray
        (x-mean(x))/std(x)
    '''
    x = np.array(x)
    if np.prod(x.shape)==0: return x
    theslice = narray.make_rebroadcast_slice(x,axis=axis,verbose=verbose)
    mx = (np.nanmean if ignore_nan else np.mean)(x,axis=axis)[theslice]
    return (x-mx)/mx

def span(data):
    '''
    Get the range of values (min,max) spanned by a dataset
    
    Parameters
    ----------
    data: np.array
    
    Returns
    -------
    span: non-negative number
        np.max(data)-np.min(data)
    '''
    data = np.array(data).ravel()
    return np.max(data)-np.min(data)

def unit_length(x,axis=0):
    '''
    Interpret given axis of multidimensional array as 
    vectors, and normalize them to unit length.
    
    Parameters
    ----------
    x : np.array
    
    Other Parameters
    ----------------
    axis : int or tuple, default None
    
    Returns
    -------
    u : np.array
        vectors in `x` normalized to unit length
    '''
    x = np.array(x)
    theslice = narray.make_rebroadcast_slice(x,axis=axis)
    return x*(np.sum(x**2,axis=axis)**-.5)[theslice]

    
def spaced_derivative(x):
    '''
    Differentiate a 1D timeseries returning a new vector 
    with the same number of samples. This smoothly 
    interpolates between a forward difference at the start 
    of the signal and a backward difference at the end of 
    the signal. 
    
    Parameters
    ----------
    x: 1D np.float32
        Signal to differentiate
    
    Returns
    -------
    '''
    N = len(x)
    return interp1d(
        np.linspace(0,1,N-1),
        np.diff(x))(
        np.linspace(0,1,N))













############################################################
# Interpolation


def upsample(x,factor=4):
    '''
    Uses fourier transform to upsample x by some factor.

    Operations:
    
    1. remove linear trend
    2. mirror to get reflected boundary
    3. take the fourier transform
    4. add padding zeros to FFT to effectively upsample
    5. taking inverse fourier transform
    6. remove mirroring
    7. restore linear tend

    Parameters
    ----------
    factor : int
        Integer upsampling factor. Default is 4.
    x : array-like
        X is cast to float64 before processing. Complex values are
        not supported.
        
    Returns
    -------
    x : array
        upsampled `x`
    '''
    assert type(factor) is int
    assert factor>1
    N = len(x)
    # Remove DC
    dc = np.mean(x)
    x -= dc
    # Remove linear trend
    dx = np.mean(np.diff(x))
    x -= np.arange(N)*dx
    # Mirror signal
    y = np.zeros(2*N,dtype=np.float64)
    y[:N]=x
    y[N:]=x[::-1]
    # Fourier transform
    ft = np.fft.fft(y).real
    # note
    # if N is even we have 1 DC and one nyquist coefficient
    # if N is odd  we have 1 DC and two nyquist-ish coefficients
    # If there is only one nyquist coefficient, it will be real-valued,
    # so it will be it's own complex conjugate, so it's fine if we
    # double it.
    up = np.zeros(N*2*factor,dtype=np.float64)
    up[:N//2+1]=ft[:N//2+1]
    up[-N//2: ]=ft[-N//2: ]
    x  = (np.fft.ifft(up).real)[:N*factor]*factor
    x += dx*np.arange(N*factor)/float(factor)
    x += dc
    return x

def nice_interp(a,b,t):
    '''
    numpy.interp1d with nice defaults
    
    Parameters
    ----------
    a: x values for interpolation
    b: y values for interpolation
    t: x values to sample at
    
    Returns
    -------
    np.array: interpolated values
    '''
    return interp1d(a,b,
        kind='cubic',fill_value=(b[0],b[-1]),bounds_error=False,axis=0)(t)












############################################################
# Correlations

def autocorrelation(x,lags=None,center=True,normalize=True):
    '''
    Computes the normalized autocorrelation over the specified
    time-lags using convolution. Autocorrelation is normalized
    such that the zero-lag autocorrelation is 1.

    TODO, fix: For long
    lags it uses FFT, but has a different normalization from 
    the time-domain implementation for short lags. In 
    practice this will not matter.
    
    Parameters
    ----------
    x : 1d array
        Data for which to compute autocorrelation function

    Other Parameters
    ----------------
    lags : int,
        Default is `min(200,len(x)`,
        Number of time-lags over which to compute the ACF. 
    center : bool, default True
        Whether to mean-center data before taking autocorrelation
    normalize : bool, default True
        Whether to normalize by zero-lag signal variance

    Returns
    -------
    ndarray
        Autocorrelation function, length 2*lags + 1
    '''
    x = np.array(x)
    if center:
        x -= np.mean(x)
    N = len(x)
    if lags is None:
        lags = min(200,N)
    # TODO: disabling FFT for now; clean up
    # For long, scalar-valued timeseries,
    # Use FFT to compute autocorrelations
    '''
    if lags>0.5*np.log2(N) and len(x.shape)==1:
        # Use FFT for long lags
        result = np.float32(fftconvolve(x,x[::-1],'same'))
        M = len(result)//2
        result *= 1./result[M]
        return result[M-lags:M+lags+1]
    '''
    # For short correlation times or 
    # vector-valued data, use iteration to compute time-lags
    # else:
    # Use time domain for short lags
    result  = np.zeros(lags*2+1,x.dtype)

    zerolag = np.var(x)
    xx = np.mean(x)**2
    result[lags] = zerolag
    #for i in range(1,lags+1):
    #    result[i+lags] = result[lags-i] = np.nanmean(x[...,i:]*x[...,:-i])
    for i in range(1,lags+1):
        result[i+lags] = result[lags-i] =\
            (np.nanmean(x[...,i:]*x[...,:-i])*(N-i)+xx*i)/N
    if normalize:
        result *= 1./zerolag
    return result
    
def fftacorr1d(x):
    '''
    Autocorrelogram via FFT.
    
    Parameters
    ----------
    x: bp.float32
    
    Returns
    -------
    '''
    x = np.float32(x)
    x = x-np.mean(x)
    N = len(x)
    x = np.concatenate([x,x[::-1]])
    a = fftshift(ifft(abs(fft(x))**2).real)[N:]
    a = a/np.max(a)
    return a
    
def zgrid(L):
    '''
    2D grid coordinates as complex numbers, 
    ranging from -L/2 to L/2
    
    Parameters
    ----------
    L: int
        Desired size of LxL grid
    
    Returns
    -------
    np.complex64: LxL coordinate grid; center is zero.
    '''
    c = np.arange(L)-L//2
    return 1j*c[:,None]+c[None,:]







    
############################################################
# Feature building

def make_lagged(x,NLAGS=5,LAGSPACE=1):
    '''
    Create shifted/lagged copies of a 1D signal.
    These are retrospective (causal) features. 
    
    Parameters
    ----------
    x: 1D np.array length `T`
    
    Other Parameters
    ----------------
    NLAGS: positive int; default 5
    LAGSPACE: positive int; default 1
    
    Returns
    -------
    result: NLAGS×T np.array
         The first element is the original unshifted signal.
         Later elements are shifts progressively further 
         back in time.
    '''
    if not len(x.shape)==1:
        raise ValueError('Signal should be one-dimensional')
    t = np.arange(len(x))
    return np.array([np.interp(t-LAG,t,x) for LAG in np.arange(NLAGS)*LAGSPACE])


def linear_cosine_basis(
    TIMERES=100,
    NBASIS=10,
    normalize=True):
    '''
    Cosine basis tiling the unit interval
    
    Other Parameters
    ----------------
    TIMERES: int; default 100
        Number of samples used to tile the unit interval.
    NBASIS: int; default 10
        Number of cosine basis functions to prepare
    normalize: boolean; default True
        Normalize sum of each basis element to 1? 
    
    Returns
    -------
    B: np.array
    '''
    times = np.linspace(0,1,TIMERES)
    bt = times*np.pi/2*(NBASIS-1)
    def cos_basis(t):
        t = np.clip(t,-np.pi,np.pi)
        return (np.cos(t)+1)*0.5
    B = np.array([
        cos_basis(bt-np.pi/2*delta) 
        for delta in np.arange(NBASIS)
    ])
    if normalize:
        B/= np.sum(B,axis=0)
    return B
    
def circular_cosine_basis(N,T):
    '''
    Periodic raised-consine basis.
    
    Parameters
    ----------
    N : number of basis functions
    T : grid resolution
    
    Returns
    -------
    B:np.array
        Periodic raised-consine basis.
    '''
    wl = 4/N
    qc = 1/N
    h      = np.linspace(0,1,T+1)[:-1]
    phases = np.linspace(0,1,N+1)[:-1]
    return 1-np.cos(np.clip((h[:,None] + phases[None,:])%1,0,wl)*2*np.pi/wl)







