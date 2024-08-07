#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Routines for multi-taper spectral analysis
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

# Suppress warnings from numpy/spectrum
import warnings
import numpy as np
import scipy.linalg

from .. import signal as sig
from neurotools.util import getfftw as fft

from neurotools.jobs.ndecorator import memoize
from neurotools.jobs            import parallel

try:
    from spectrum.mtm import dpss
except:
    warnings.warn("Could not find the `spectrum` module; multitaper unsupported")
    def dpss(*args):
        raise NotImplementedError("Please install the spectrum module, e.g.\n\tpip install spectrum")

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


def spectrum(x,k,Fs=1000.0,nodc=True,return_negative=False):
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
        average squared power over tapers.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        N = np.shape(x)[-1]
        if nodc:
            x = x-np.mean(x,axis=-1)[...,None]
        tapers, eigen = dpss_cached(N,0.49999*k)
        eigen /= np.sum(eigen)
        specs = [np.abs(fft.fft(x*t))**2 for t in tapers]
        freqs = fft.fftfreq(N,1./Fs)
        psd = eigen@specs
        if return_negative:
            return freqs,psd
        else:
            return freqs[:N//2],psd[...,:N//2]


def population_coherence(
    x,y,FS,
    lowf=0,
    highf=None,
    k=None):
    '''
    Computes coherence spectrum between two collections of signals.
    Uses multitaper averaging.
    For each frequency, computes a pairwise matrix of coherence between
    both collections of signals.
    Returns the sum of the singular values of this coherence matrix
    as a summary of population coherence.
    '''

    x = np.array(x)
    y = np.array(y)
    
    if len(x.shape)==1: x=x.reshape(1,len(x))
    if len(y.shape)==1: y=y.reshape(1,len(y))

    # Check arguments
    if not len(x.shape)==2:
        raise ValueError('Input arrays should be Nfeature x Ntime in shape')
    T = x.shape[1]
    if not y.shape[1]==T:
        raise ValueError('Both sets of signals should have same No. timepoints')
    if k is None:
        k = max(5,int(round(T*lowf)))
    if highf is None:
        highf = FS*0.49

    # Z-score and band-limit signals
    x = np.array([sig.zscore(sig.bandpass_filter(z,lowf,None,FS)) for z in x])
    y = np.array([sig.zscore(sig.bandpass_filter(z,lowf,None,FS)) for z in y])

    tapers,taper_evals = dpss_cached(T,(1/2-1e-9)*k)
    
    # Limit freqencies used to conserve memory
    freqs = fft.fftfreq(T,1./FS)
    use   = (freqs<=highf)&(freqs>=lowf)
    freqs = freqs[use]

    # Compute tapered power density estimates for all signals
    # Also compute tapered cross-spectral estiamtes
    result = []
    for tf,e in zip(tapers,taper_evals):
        ftx = np.array([fft.fft(z*tf)[use] for z in x])
        fty = np.array([fft.fft(z*tf)[use] for z in y])
        pxx = np.abs(ftx)**2
        pyy = np.abs(fty)**2
        pxy = np.abs(ftx[:,None,:]*np.conj(fty[None,:,:]))**2
        result.append((pxx,pyy,pxy))
    pxx,pyy,pxy = zip(*result)

    # Compute power averaged over tapers, then compute coherence    
    pxx = np.mean(pxx,axis=0)
    pyy = np.mean(pyy,axis=0)
    pxy = np.mean(pxy,axis=0)
    coherence = pxy/(pxx[:,None,:]*pyy[None,:,:])
    return freqs,coherence




def _tapered_cross_specra_helper(params):
    '''
    '''
    i,(x,y,use,taper,e) = params
    ftx = np.array([fft.fft(z*taper)[use] for z in x])
    fty = np.array([fft.fft(z*taper)[use] for z in y])
    pxx = np.abs(ftx)**2*e
    pyy = np.abs(fty)**2*e
    pxy = np.abs(ftx[:,None,:]*np.conj(fty[None,:,:]))**2*e
    result = (pxx,pyy,pxy)
    return i,result


def population_eigencoherence(
    x,y,FS,
    lowf=0,
    highf=None,
    k=None,
    use_parallel=False
    ):
    '''
    Computes coherence spectrum between two collections of signals.
    Uses multitaper averaging.
    For each frequency, computes a pairwise matrix of coherence between
    both collections of signals.
    Returns the sum of the singular values of this coherence matrix
    as a summary of population coherence.
    '''

    x = np.array(x)
    y = np.array(y)
    
    if len(x.shape)==1: x=x.reshape(1,len(x))
    if len(y.shape)==1: y=y.reshape(1,len(y))

    # Check arguments
    if not len(x.shape)==2:
        raise ValueError('Input arrays should be Nfeature x Ntime in shape')
    T = x.shape[1]
    if not y.shape[1]==T:
        raise ValueError('Both sets of signals should have same No. timepoints')
    if k is None:
        k = max(5,int(round(T*lowf)))
    if highf is None:
        highf = FS*0.49

    # Z-score and band-limit signals
    x = np.array([sig.zscore(sig.bandpass_filter(z,lowf,None,FS)) for z in x])
    y = np.array([sig.zscore(sig.bandpass_filter(z,lowf,None,FS)) for z in y])

    tapers,taper_evals = dpss_cached(T,(1/2-1e-9)*k)

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

