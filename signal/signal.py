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

import numpy as np
from scipy.signal.signaltools import fftconvolve,hilbert
from scipy.signal import butter, filtfilt, lfilter

from   pylab import find

def geometric_window(c,w):
    '''
    Gemoetrically center a window in frequency
    
    Parameters
    ----------
    c : float
        Center of frequency window
    w : float
        width of frequency window
        
    Returns
    -------
    fa : float
        low-frequency cutoff
    fb : float
        high-frequency cutoff
    '''
    if not c>0:
        raise ValueError('The center of the window should be positive')
    if not w>=0:
        raise ValueError('The window size should be non-negative')
    lgwindow = (w+np.sqrt(w**2+4*c**2))/(2*c)
    fa       = c/lgwindow
    fb       = c*lgwindow
    return fa,fb

def gaussian_kernel(sigma):
    '''
    generate 1D Guassian kernel for smoothing
    sigma: standard deviation, >0
    
    Parameters
    ----------
    Returns
    -------
    '''
    assert sigma>0
    K = np.ceil(sigma)
    N = K*2+1
    K = np.exp( - (np.arange(N)-K)**2 / (2*sigma**2) )
    K *= 1./np.sum(K)
    return K

def gaussian_smooth(x,sigma):
    '''
    Smooth signal x with gaussian of standard deviation sigma

    sigma: standard deviation
    x: 1D array-like signal
    
    Parameters
    ----------
    Returns
    -------
    '''
    K = gaussian_kernel(sigma)
    return np.convolve(x,K,'same')

def zscore(x,axis=0,regularization=1e-30):
    '''
    Z-scores data, defaults to the first axis.
    A regularization factor is added to the standard deviation to preven
    numerical instability when the standard deviation is extremely small.
    The default refularization is 1e-30.
    x: NDarray
    axis: axis to zscore; default 0
    
    Parameters
    ----------
    x
        array-like real-valued signal
    Returns
    -------
    x
        (x-mean(x))/std(x)
    '''
    ss = np.std(x,axis=axis)+regularization
    return (x-np.mean(x,axis=axis))/ss


def unitscale(signal):
    '''
    Rescales `signal` so that its minimum is 0 and its maximum is 1.

    Parameters
    ----------
    signal
        array-like real-valued signal
    Returns
    -------
    signal
        Rescaled signal-min(signal)/(max(signal)-min(signal))
    '''
    signal = np.array(signal)
    signal-= np.min(signal)
    signal/= np.max(signal)
    return signal

def local_maxima(x):
    '''
    Returns signal index and values at those indecies

    Parameters
    ----------
    Returns
    -------
    '''
    t = find(np.diff(np.sign(np.diff(x)))<0)+1
    return t,x[t]

def local_minima(x):
    '''
    Returns signal index and values at those indecies for all local minima.
    See local_maxima

    Parameters
    ----------
    Returns
    -------
    '''
    t,x = local_maxima(-x)
    return t,-x

def amp(x):
    '''
    Extracts amplitude envelope using Hilbert transform. X must be narrow
    band. No padding is performed so watch out for boundary effects
    
    Parameters
    ----------
    Returns
    -------
    '''
    return abs(hilbert(x))

def getsnips(signal,times,window):
    '''
    Extract snippits of a time series surronding a list of times. Typically
    used for spike-triggered statistics
    
    Parameters
    ----------
    Returns
    -------
    '''
    times = times[times>window]
    times = times[times<len(signal)-window-1]
    snips = np.array([signal[t-window:t+window+1] for t in times])
    return snips

def triggeredaverage(signal,times,window):
    '''
    
    Parameters
    ----------
    Returns
    -------
    '''
    return np.mean(getsnips(signal,times,window),0)

def gettriggeredstats(signal,times,window):
    '''
    Get a statistical summary of data in length window around time point
    times.
    
    Parameters
    ----------
    signal : one-dimensional array-like
        Signal to summarize
    times : one-dimensionan array-like
        list of time-points around which to summarize (in frames)
    window : positive int
        window (in frames) around each time-point to use for statistical
        summary
        
    Returns
    -------
    means : 
        means of `signal` for all time-windows specified in `times`
    standard-deviations : 
        std of `signal` for all time-windows specified in `times`
    standard-erros : 
        standard errors of the mean of `signal` for all time-windows 
        specified in `times`
    '''
    s = getsnips(signal,times,window)
    return np.mean(s,0),np.std(s,0),np.std(s,0)/np.sqrt(len(times))*1.96

def padout(data):
    '''
    Generates a reflected version of a 1-dimensional signal. This can be
    handy for achieving reflected boundary conditions in algorithms that
    do not support this condition by default.

    The original data is placed in the middle, between the mirrord copies.
    Use the function "padin" to strip the padding
    
    Parameters
    ----------
    Returns
    -------
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

def nonnegative_bandpass_filter(data,fa=None,fb=None,
    Fs=1000.,order=4,zerophase=True,bandstop=False,
    offset=1.0):
    '''
    For filtering data that must remain non-negative. Due to ringing
    conventional fitering can create values less than zero for non-
    negative real inputs. This may be unrealistic for some data.

    To compensate, this performs the filtering on the natural
    logarithm of the input data. For small numbers, this can lead to
    numeric underflow, so an offset parameter (default 1) is added
    to the data for stability.

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
    order (1..6): 
        butterworth filter order. Default 4
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
    Array can be any dimension, filtering performed over last dimension

    Args:
        data (ndarray): data, filtering performed over last dimension
        fa (number): low-frequency cutoff. If none, highpass at fb
        fb (number): high-frequency cutoff. If none, lowpass at fa
        order (1..6): butterworth filter order. Default 4
        zerophase (boolean): Use forward-backward filtering? (true)
        bandstop (boolean): Do band-stop rather than band-pass
    
    Parameters
    ----------
    Returns
    -------
    '''
    N = data.shape[-1]
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
        b,a  = butter(order,fb/(0.5*Fs),btype='low')
        assert not bandstop
    else: raise Exception('Both fa and fb appear to be None')
    return (filtfilt if zerophase else lfilter)(b,a,padded)[...,N//2:N//2+N]
    assert 0

'''For backward compatibility, bandpass_filter is aliased as bandfilter'''
bandfilter = bandpass_filter

def box_filter(data,smoothat):
    '''
    Smooths data by convolving with a size smoothat box
    provide smoothat in units of frames i.e. samples (not ms or seconds)
    
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
    N = len(data)
    assert len(data.shape)==1
    padded = np.zeros(2*N,dtype=data.dtype)
    padded[N//2:N//2+N]=data
    padded[:N//2]=data[N//2:0:-1]
    padded[N//2+N:]=data[-1:N//2-1:-1]
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
    n = x.shape[0]
    if mode=='valid':
        filtered = [np.median(x[i:i+window]) for i in range(n-window)]
        return array(filtered)
    if mode=='same':
        a = window // 2
        b = window - a
        filtered = [np.median(x[max(0,i-a):min(n,i+b)]) for i in range(n)]
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

def rewrap(x):
    '''
    Used to handle wraparound when getting phase derivatives.
    See pdiff.
    
    Parameters
    ----------
    Returns
    -------
    '''
    x = np.array(x)
    return (x+np.pi)%(2*np.pi)-np.pi

def pdiff(x):
    '''
    Take the derivative of a sequence of phases.
    Times when this derivative wraps around form 0 to 2*np.pi are correctly
    handeled.
    
    Parameters
    ----------
    Returns
    -------
    '''
    x = np.array(x)
    return rewrap(np.diff(x))

def pghilbert(x):
    '''
    Extract phase gradient using the hilbert transform. See also pdiff.
    
    Parameters
    ----------
    Returns
    -------
    '''
    x = np.array(x)
    return pdiff(np.angle(np.hilbert(x)))

def fudge_derivative(x):
    '''
    Discretely differentiating a signal reduces its signal by one sample.
    In some cases, this may be undesirable. It also creates ambiguity as
    the sample times of the differentiated signal occur halfway between the
    sample times of the original signal. This procedure uses averaging to
    move the sample times of a differentiated signal back in line with the
    original.
    
    Parameters
    ----------
    Returns
    -------
    '''
    n = len(x)+1
    result = np.zeros(n)
    result[1:]   += x
    result[:-1]  += x
    result[1:-1] *= 0.5
    return result

def ifreq(x,Fs=1000,mode='pad'):
    '''
    Extract the instantaneous frequency from a narrow-band signal using
    the Hilbert transform.
    
    Parameters
    ----------
    Fs : int
        defaults to 1000
    mode : str
        'pad' will return a signal of the original length
        'valid' will return a signal 1 sample shorter, with derivative
        computed between each pair od points in the original signal.
    '''
    pg = pghilbert(x)
    pg = pg/(2*np.pi)*Fs
    if mode=='valid':
        return pg # in Hz
    if mode=='pad':
        return fudge_derivative(pg)
    assert 0

def unwrap(h):
    '''
    Unwraps a sequences of phase measurements so that rather than
    ranging from 0 to 2*np.pi, the values increase (or decrease) continuously.
    
    Parameters
    ----------
    Returns
    -------
    '''
    d = fudgeDerivative(pdiff(h))
    return np.cumsum(d)

def ang(x):
    '''
    Uses the Hilbert transform to extract the phase of x. X should be
    narrow-band. The signal is not padded, so be wary of boundary effects.
    
    Parameters
    ----------
    Returns
    -------
    '''
    return np.angle(np.hilbert(x))

def randband(N,fa=None,fb=None,Fs=1000):
    '''
    Returns Gaussian random noise band-pass filtered between fa and fb.
    
    Parameters
    ----------
    Returns
    -------
    '''
    return zscore(bandfilter(np.random.randn(N*2),fa=fa,fb=fb,Fs=Fs))[N//2:N//2+N]

def arenear(b,K=5):
    '''
    Expand a boolean/binary sequence by K samples in each direction.
    See "aresafe"
    
    Parameters
    ----------
    Returns
    -------
    '''
    for i in range(1,K+1):
        b[i:] |= b[:-i]
    for i in range(1,K+1):
        b[:-i] |= b[i:]
    return b

def aresafe(b,K=5):
    '''
    Contract a boolean/binary sequence by K samples in each direction.
    For example, you may want to test for a condition, but avoid samples
    close to edges in that condition.
    
    Parameters
    ----------
    Returns
    -------
    '''
    for i in range(1,K+1):
        b[i:] &= b[:-i]
    for i in range(1,K+1):
        b[:-i] &= b[i:]
    return b

def get_edges(signal):
    '''
    Assuming a binary signal, get the start and stop times of each
    treatch of "1s"
    
    Parameters
    ----------
    Returns
    -------
    '''
    if len(signal)<1:
        return np.array([[],[]])
    starts = list(find(np.diff(np.int32(signal))==1))
    stops  = list(find(np.diff(np.int32(signal))==-1))
    if signal[0 ]: starts = [0]+starts
    if signal[-1]: stops = stops + [len(signal)]
    return np.array([starts, stops])

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

def phase_rotate(s,f,Fs=1000.):
    '''
    Only the phase advancement portion of a resonator.
    See resonantDrive
    
    Parameters
    ----------
    Returns
    -------
    '''
    theta = f*2*np.pi/Fs
    s *= np.exp(1j*theta)
    return s

def fm_mod(freq):
    '''
    
    Parameters
    ----------
    Returns
    -------
    '''
    N = len(freq)
    signal = [1]
    for i in range(N):
        signal.append(phaseRotate(signal[-1],freq[i]))
    return np.array(signal)[1:]

def pieces(x,thr=4):
    '''
    
    Parameters
    ----------
    Returns
    -------
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

def stats_block(data,statfunction,N=100):
    '''
    Compute function of signal in blocks of size $N$ over the last axis
    of the data
    
    Parameters
    ----------
    data : np.array
        N-dimensional numpy array. Blocking is performed over the last axis
    statfunction : function
        Statistical function to compute on each block. Should be, or 
        behave similarly to, the `numpy` buit-ins, e.g. np.mean,
        np.median, etc.
    N : positive integer
        Block size in which to break data. If data cannot be split 
        evenly into blocks of size $N$, then data are truncated to the 
        largest integer multiple of N. 
    
    Returns
    -------
    np.array : 
        Blocked data
    '''
    N = int(N)
    L = data.shape[-1]
    B = L//N
    D = B*N
    data = data[...,:D]
    data = np.reshape(data,data.shape[:-1]+(B,N))
    return statfunction(data,axis=-1)

def mean_block(data,N=100):
    '''
    Calls stats_block using np.mean. See documentation of stats_block for
    details.
    
    Parameters
    ----------
    Returns
    -------
    '''
    return stats_block(data,np.mean,N)

def var_block(data,N=100):
    '''
    Calls stats_block using np.var. See documentation of stats_block for
    details.
    
    Parameters
    ----------
    Returns
    -------
    '''
    return stats_block(data,np.var,N)

def median_block(data,N=100):
    '''
    Calls stats_block using np.median. See documentation of stats_block for
    details.
    
    Parameters
    ----------
    Returns
    -------
    '''
    return stats_block(data,np.median,N)

def phase_randomize(signal):
    '''
    Phase randomizes a signal by rotating frequency components by a random
    angle. Negative frequencies are rotated in the opposite direction.
    The nyquist frequency, if present, has it's sign randomly flipped.
    
    Parameters
    ----------
    Returns
    -------
    '''
    assert 1==len(signam.shape)
    N = len(signal)
    if N%2==1:
        # signal length is odd.
        # ft will have one DC component then symmetric frequency components
        randomize  = np.exp(1j*np.random.rand((N-1)/2))
        conjugates = np.conj(randomize)[::-1]
        randomize  = np.append(randomize,conjugates)
    else:
        # signal length is even
        # will have one single value at the nyquist frequency
        # which will be real and can be sign flipped but not rotated
        flip = 1 if rand(1)<0.5 else -1
        randomize  = np.exp(1j*rand((N-2)/2))
        conjugates = np.conj(randomize)[::-1]
        randomize  = np.append(randomize,flip)
        randomize  = np.append(randomize,conjugates)
    # the DC component is not randomized
    randomize = np.append(1,randomize)
    # take FFT and apply phase randomization
    ff = fft(signal)*randomize
    # take inverse
    randomized = ifft(ff)
    return real(randomized)

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
        rephase = np.exp(1j*2*np.pi*rand((N-2)/2))
        rephase = np.concatenate([rephase,[sign(rand()-0.5)],conj(rephase[::-1])])
    else: # N is odd
        rephase = np.exp(1j*2*np.pi*rand((N-1)/2))
        rephase = np.append(rephase,np.conj(rephase[::-1]))
    rephase = np.append([1],rephase)
    x *= rephase
    return np.real(ifft(x))

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
    return bandfilter(x,fb=cut,Fs=fs,order=order)

def highpassFilter(x, cut=40, Fs=1000, order=4):
    '''
    Execute a butterworth high pass Filter at frequency "cut"
    Defaults to order=4 and Fs=1000
    
    Parameters
    ----------
    Returns
    -------
    '''
    return bandfilter(x,fa=cut,Fs=Fs,order=order)

def fdiff(x,Fs=240.):
    '''
    Take the discrete derivative of a signal, correcting result for
    sample rate. This procedure returns a singnal two samples shorter than
    the original.
    
    Parameters
    ----------
    Returns
    -------
    '''
    return (x[2:]-x[:-2])*Fs*.5

def killSpikes(x,threshold=1):
    '''
    Remove times when the signal exceeds a given threshold of the
    standard deviation of the underlying signal. Removed data are
    re-interpolated from the edges. This procedure is particularly
    useful in correcting kinematics velocity trajectories. Velocity
    should be smooth, but motion tracking errors can cause sharp spikes
    in the signal.
    
    Parameters
    ----------
    Returns
    -------
    '''
    x = np.array(x)
    y = zscore(highpassFilter(x))
    x[abs(y)>threshold] = nan
    for s,e in zip(*get_edges(isnan(x))):
        a = x[s-1]
        b = x[e+1]
        print(s,e,a,b)
        x[s:e+2] = np.linspace(a,b,e-s+2)
    return x

def peak_within(freqs,spectrum,fa,fb):
    '''
    Find maximum within a band
    
    Parameters
    ----------
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

def local_peak_within(freqs,cc,fa,fb):
    '''
    For a spectrum, identify the largest local maximum in the frequency
    range [fa,fb].
    
    Parameters
    ----------
    Returns
    -------
    '''
    local = local_maxima(cc)[0]
    peaks = list(set(local) & set(find((freqs>=fa) & (freqs<=fb))))
    if len(peaks)==0: return (None,)*3 # no peaks!
    i     = peaks[argmax(cc[peaks])]
    return i, freqs[i], cc[i]

def zeromean(x,axis=None):
    '''
    Remove the mean trend from data
    
    Parameters
    ----------
    Returns
    -------
    '''
    return x-np.mean(x,axis=axis)

def sign_preserving_amplitude_demodulate(analytic_signal,doplot=False):
    '''
    Extracts an amplitude-modulated component from an analytic signal,
    Correctly flipping the sign of the signal when it crosses zero,
    rather than returning a rectified result.

    Sign-changes are heuristically detected basd on the following:
        - An abnormally large skip in phase between two time points,
          larger than np.pi/2, that is also a local extremum in phase velocity
        - local minima in the amplitude at low-voltage with high curvature

    
    Parameters
    ----------
    Returns
    -------
    '''

    analytic_signal = zscore(analytic_signal)

    phase      = np.angle(analytic_signal)
    amplitude  = np.abs(analytic_signal)

    phase_derivative     = fudge_derivative(pdiff(phase))
    phase_curvature      = fudge_derivative(np.diff(phase_derivative))
    amplitude_derivative = fudge_derivative(np.diff(amplitude))
    amplitude_curvature  = fudge_derivative(np.diff(amplitude_derivative))

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


def autocorrelation(x,lags=None):
    '''
    Computes the normalized autocorrelation over the specified
    time-lags using convolution. Autocorrelation is normalized
    such that the zero-lag autocorrelation is 1.

    This was written in haste and needs some work. For long
    lags it uses FFT, but has a different normalization from the
    time-domain implementation for short lags. In practice this
    will not matter, but formally it's good to be rigorous.

    Parameters
    ----------
    x : 1d array
        Data for which to compute autocorrelation function
    lags : int
        Number of time-lags over which to compute the ACF. Default
        is min(200,len(x))

    Returns
    -------
    ndarray
        Autocorrelation function, length 2*lags + 1
    '''
    x = np.array(x)
    x -= np.mean(x)
    N = len(x)
    if lags is None:
        lags = min(200,N)
    # TODO: TUNE THIS CONSTANT
    if lags>0.5*np.log2(N):
        # Use FFT for long lags
        result = np.float32(fftconvolve(x,x[::-1],'same'))
        M = len(result)//2
        result *= 1./result[M]
        return result[M-lags:M+lags+1]
    else:
        # Use time domain for short lags
        result  = np.zeros(lags*2+1,'float')
        zerolag = np.var(x)
        result[lags] = zerolag
        for i in range(1,lags):
            result[i+lags] = result[lags-i] = np.mean(x[i:]*x[:-i])
        result *= 1./zerolag
        return result


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
    '''
    assert type(factor) is int
    assert factor>1
    N = len(x)
    dc = np.mean(x)
    x -= dc
    dx = np.mean(np.diff(x))
    x -= np.arange(N)*dx
    y = np.zeros(2*N,dtype=float64)
    y[:N]=x
    y[N:]=x[::-1]
    ft = fft(y)
    # note
    # if N is even we have 1 DC and one nyquist coefficient
    # if N is odd we have 1 DC and two nyquist coefficients
    # If there is only one nyquist coefficient, it will be real-values,
    # so it will be it's own complex conjugate, so it's fine if we
    # double it.
    up = np.zeros(N*2*factor,dtype=float64)
    up[:N//2+1]=ft[:N//2+1]
    up[-N//2:] =ft[-N//2:]
    x = (ifft(up).real)[:N*factor]*factor
    x += dx*np.arange(N*factor)/float(factor)
    x += dc
    return x

def linfilter(A,C,x,initial=None):
    '''
    Linear response filter on data $x$ for system
    
    $$
    \partial_t z = A z + C x(t)
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
