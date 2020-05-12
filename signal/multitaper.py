#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import neurotools.getfftw as fft
from neurotools.signal import zscore
from neurotools.jobs.ndecorator import memoize

try:
    from spectrum.mtm import dpss
except:
    def dpss(*args):
        raise NotImplementedError("Please install the spectrum module, e.g.\n\tpip install spectrum")

# Suppress warnings from numpy/spectrum
import warnings

@memoize
def dpss_cached(length,half_bandwidth_parameter):
    '''
    Get a collection of DPSS tapers. The number of tapers equals the
    half bandwidth parameter times two. For legacy reasons the tapers
    are returned transposed such that the first dimension indexes
    tapers rather than time.

    The advantage of using this function is that computing DPSS is
    expensive. This function caches the results in RAM.

    Parameters
    ----------
    length : integer
        length of the domain for which to compute the DPSS
    half_bandwidth_parameter : number
        The number of is the half_bandwidth_parameter*2

    Returns
    -------
    ndarray:
        tapers.T a transposed list of DPSS tapers
    ndarray:
        taper eigenvalues ( weights )
    '''
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        tapers,eigen = dpss(int(length),half_bandwidth_parameter)
    return tapers.T,eigen

def multitaper_spectrum(x,k,Fs=1000.0,nodc=True):
    '''
    Parameters
    ----------
    x : ndarray
        Signal to use; spectrum taken over last dimension
    k : int (positive)
        number of tapers (positive)
    Fs: int
        sample rate in Hz (default 1000)

    Returns
    -------
    ndarray
        frequencies,
    ndarray
        average sqrt(power) over tapers.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        N = x.shape[-1]
        if nodc:
            x = x-np.mean(x,axis=-1)[...,None]
        tapers, eigen = dpss_cached(N,0.4999*k)
        specs = [np.abs(fft.fft(x*t)) for t in tapers]
        freqs = fft.fftfreq(N,1./Fs)
        return freqs[:N//2],np.mean(specs,0)[...,:N//2]

def multitaper_squared_spectrum(x,k,Fs=1000.0,nodc=True):
    '''
    Parameters
    ----------
    x : ndarray
        Signal to use
    k : int (positive)
        number of tapers (positive)
    Fs: int
        sample rate in Hz (default 1000)

    Returns
    -------
    ndarray
        frequencies,
    ndarray
        average squared power over tapers.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        N = np.shape(x)[-1]
        if nodc:
            x = x-np.mean(x,axis=-1)[...,None]
        tapers, eigen = dpss_cached(N,0.4999*k)
        specs = [np.abs(fft.fft(x*t)) for t in tapers]
        freqs = fft.fftfreq(N,1./Fs)
        return freqs[:N//2],np.mean(specs,0)[...,:N//2]**2

def sliding_multitaper_spectrum(x,window=500,step=100,Fs=1000,BW=5):
    '''
    NOT IMPLEMENTED
    '''
    raise NotImplementedError("This function is not yet implemented")


from neurotools.signal import zscore, bandpass_filter
import scipy.linalg

def _tapered_cross_specra_helper(params):
    i,(x,y,use,taper,e) = params
    ftx = np.array([fft.fft(z*taper)[use] for z in x])
    fty = np.array([fft.fft(z*taper)[use] for z in y])
    pxx = np.abs(ftx)*e
    pyy = np.abs(fty)*e
    pxy = np.abs(ftx[:,None,:]*np.conj(fty[None,:,:]))*e
    result = (pxx,pyy,pxy)
    return i,result

from neurotools.jobs import parallel

def multitaper_population_eigencoherence(
    x,y,FS,
    lowf=0,
    highf=None,
    NTAPER=None,
    use_parallel=False):
    '''
    Computes coherence spectrum between two collections of signals.
    Uses multitaper averaging.
    For each frequency computes a pairwise matrix of coherence between
    both collections of signals.
    Returns the sum of the singular values of this coherence matrix
    as a summary of population coherence.
    '''

    x = np.array(x)
    y = np.array(y)

    # Check arguments
    if not len(x.shape)==2:
        raise ValueError('Input arrays should be Nfeature x Ntime in shape')
    T = x.shape[1]
    if not y.shape[1]==T:
        raise ValueError('Both sets of signals should have same No. timepoints')
    if NTAPER is None:
        NTAPER = max(5,int(round(T*lowf)))
    if highf is None:
        highf = FS*0.49

    # Z-score and band-limit signals
    x = np.array([zscore(bandpass_filter(z,lowf,None,FS)) for z in x])
    y = np.array([zscore(bandpass_filter(z,lowf,None,FS)) for z in y])

    tapers,taper_evals = dpss_cached(T,(1/2-1e-9)*NTAPER)

    if use_parallel:
        parallel.reset_pool()

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Limit freqencies used to conserve memory
        freqs = fft.fftfreq(T,1./FS)
        use   = (freqs<=highf)&(freqs>=lowf)
        freqs = freqs[use]

        # Compute tapered power density estimates for all signals
        # Also compute tapered cross-spectral estiamtes
        problems = [(x,y,use,taper,e) for taper,e in zip(tapers,taper_evals)]
        result = parallel.parmap(_tapered_cross_specra_helper,enumerate(problems),debug=use_parallel)
        pxx,pyy,pxy = zip(*result)

    # Compute power averaged over tapers, then compute coherence    
    pxx = np.mean(pxx,axis=0)
    pyy = np.mean(pyy,axis=0)
    pxy = np.mean(pxy,axis=0)
    coherence = pxy/(pxx[:,None,:]*pyy[None,:,:])

    # Get sum of singular values for each frequency
    ecohere = np.array(parallel.parmap(_eigencoherence_helper,enumerate(coherence.T),debug=use_parallel))
    #ecohere = np.array([np.sum(scipy.linalg.svd(c)[1]) for c in coherence.T])

    if use_parallel:
        parallel.reset_pool()

    return freqs,ecohere

def _eigencoherence_helper(params):
    i,c = params
    return i,np.sum(scipy.linalg.svd(c)[1]) 

