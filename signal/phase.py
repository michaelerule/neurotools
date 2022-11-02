#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Routines for working with analytic signals.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
from scipy.signal import hilbert

import neurotools.signal as sig
from neurotools.util.array import find

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
    np.array:
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
    return sig.zscore(
            sig.bandfilter(
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
    :np.array
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

    Sign-changes are heuristically detected basd on the following:
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
    demodulated: np.array
    '''

    analytic_signal = sig.zscore(analytic_signal)

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

