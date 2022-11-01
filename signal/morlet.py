#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Coherence measures based on the Morlet wavelength. 

This allows adjusting the spectral and temporal nfreqss
depending on frequency. 
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function


from neurotools.graphics.plot   import *
from neurotools.util.getfftw    import *
from neurotools.jobs.ndecorator import memoize
from scipy.signal import morlet

def normalized_morlet(m,w):
    '''
    See morlet(m,w)

    This applies post-processing such that the sum absolute magnitued of
    the wavelet is 1
    '''
    wl = morlet(int(m),w)
    return wl/sum(abs(wl))

@memoize
def prepare_wavelet_fft_basis(fa,fb,nfreqs,L,w,Fs=1000):
    '''
    Parameters
    ----------
    fa: positive float
        Low-frequency cutoff in Hz
    fb: positive float
        High-frequency cutoff in Hz
    nfreqs: positive int
        Specral sampling nfreqs
    L: positive int
        Number of samples
    w: float
        Base frequency,
        passed as second argument to `morlet()`
    
    Other Parameters
    ----------------
    Fs: positive int; default 1000
        Sampling rate
    
    Returns
    -------
    freqs:
        FFT frequencies
    basis: 
        Fourier transform of Morlet wavelets for each
        band.
    '''    
    freqs = arange(fa,fb,nfreqs)
    M     = 2.*1.*w*Fs/freqs
    # we have to be careful about padding. In general we want the wavelets
    # to all be centered in the array before we take the FFT, so the padding
    # should be symmetric
    #
    # Modification: now normalizing wavelet amplitude to 1
    allwl = []
    fftwl = []
    for i,m in enumerate(M):
        wl = morlet(m,w)
        wl = wl/sum(abs(wl))
        N = len(wl)
        if N>L:
            chop_begin = (N-L)/2
            chop_end   = (N-L)-chop_begin
            wl = wl[chop_begin:-chop_end]
            N = len(wl)
            assert N==L
        padded = zeros(L,dtype=complex64)
        start  = (L-N)/2
        padded[start:start+N]=wl
        reordered = zeros(L,dtype=complex64)
        reordered[L/2:]=padded[:L/2]
        reordered[:L/2]=padded[L/2:]
        allwl.append(reordered)
        fftwl.append(fft(reordered))
    return freqs,array(fftwl)

def fft_cwt(data,fa,fb,w=4.0,nfreqs=0.1,Fs=1000):
    '''
    Wavelet transform.

    Parameters
    ----------
    data: 
        NTIMES × NCHANNELS np.array
    
    Other Parameters
    ----------------
    w: positive float; default 4.0
        Wavelet base frequency; 
        Controls the time-frequency tradeoff
    nfreqs: positive float; default 0.1
        Frequency sampling nfreqs
    Fs: positive int; default 1000
        Sample rate
    
    Returns
    -------
    freqs: 
        Frequencies of each band, in Hz
    result: NCHANNELS × NFREQS × NTIMES np.array
        wavelet transform
    '''    
    if len(shape(data))==1:
        data = reshape(data,shape(data)+(1,))
    N,NCH        = shape(data)
    if NCH>N:
        print('MORE CHANNELS THAN DATA CHECK FOR TRANSPOSED')
    padded       = zeros((N*2,NCH),dtype=complex64)
    padded[:N,:] = data
    padded[N:,:] = data[::-1,:]
    fft_data = fft(padded,axis=0)
    freqs,wavelets = prepare_wavelet_fft_basis(fa,fb,nfreqs,N*2,w,Fs)
    result   = array([ifft(fft_data.T*wl,axis=1)[:,:N] for wl in wavelets])
    return freqs,transpose(result,(1,0,2))
    
    
def logfreqs(fa,fb,nfreq):
    '''
    Parameters
    ----------
    fa: positive float
        Low-frequency cutoff in Hz
    fb: positive float
        High-frequency cutoff in Hz
    nfreq: positive int
        Number of frequency bands
    
    Returns
    -------
    freqs: np.array
        list of logarithmically-spaced frequencies between
        `fa` and `fb`.
    '''    
    freqs = 2**np.linspace(np.log2(fa),np.log2(fb),nfreq)
    return freqs

@memoize
def prepare_wavelet_fft_basis_logspace(fa,fb,nfreq,L,w,Fs=1000):
    '''
    Parameters
    ----------
    fa: positive float
        Low-frequency cutoff in Hz
    fb: positive float
        High-frequency cutoff in Hz
    nfreq: positive int
        Number of frequency bands
    L: positive int
        Number of samples
    w: float
        Base frequency,
        passed as second argument to `morlet()`
    
    Other Parameters
    ----------------
    Fs: positive int; default 1000
        Sampling rate
    
    Returns
    -------
    freqs:
        FFT frequencies
    basis: 
        Fourier transform of Morlet wavelets for each
        band.
    '''
    freqs = logfreqs(fa,fb,nfreq)
    M     = 2.*1.*w*Fs/freqs
    # we have to be careful about padding. In general we want the wavelets
    # to all be centered in the array before we take the FFT, so the padding
    # should be symmetric
    # Modification: now normalizing wavelet amplitude to 1
    allwl = []
    fftwl = []
    for i,m in enumerate(M):
        wl = normalized_morlet(m,w)
        N = len(wl)
        if N>L:
            chop_begin = (N-L)/2
            chop_end   = (N-L)-chop_begin
            wl = wl[chop_begin:-chop_end]
            N = len(wl)
            assert N==L
        padded = np.zeros(L,dtype=np.complex64)
        start  = int((L-N)//2)
        padded[start:start+N]=wl
        reordered = np.zeros(L,dtype=np.complex64)
        reordered[L//2:]=padded[:L//2]
        reordered[:L//2]=padded[L//2:]
        allwl.append(reordered)
        fftwl.append(fft(reordered))
    return freqs,np.array(fftwl)


def population_synchrony_spectrum(
    lfp,fa,fb,w=4.0,nfreqs=0.1,Fs=1000):
    '''
    
    Use Morlet wavelets to compute short-timescale synchrony.

    Parameters
    ----------
    lfp: np.array
        First dimension is nchannels, second is time.
    
    Other Parameters
    ----------------
    w: positive float; default 4.0
        Wavelet base frequency; 
        Controls the time-frequency tradeoff
    nfreqs: positive float; default 0.1
        Frequency sampling nfreqs
    Fs: positive int; default 1000
        Sample rate
    
    Returns
    -------
    freqs: NFREQS np.array
        Frequencies of each band, in Hz
    synchrony:
        NCHANNELS × NFREQS × NTIMES np.array
    '''
    assert 0
    # this measures synchrony not coherence!
    freqs, transformed = fft_cwt(lfp.T,fa,fb,w,nfreqs,Fs)
    coherence = abs(mean(transformed,0))/mean(abs(transformed),0)
    return freqs, coherence
    
    
def fft_cwt_transposed(
    data,fa,fb,
    w=4.0,nfreqs=0.1,Fs=1000.0,threads=1):
    '''
    Parameters
    ----------
    data : numeric
        NCH x Ntimes list of signals to transform

    Returns
    ------
    freqs : float
        frequencies
    result : wavelt transforms
        Nch x Nfreq x Ntimes
    '''
    fa,fb = map(float,(fa,fb))
    if len(np.shape(data))==1:
        data = data[None,...]
    NCH,N = np.shape(data)
    if NCH>N:
        warn('MORE CHANNELS THAN DATA, CHECK IF TRANSPOSED')
    padded       = np.zeros((NCH,N*2),dtype=np.complex64)
    padded[:,:N] = data
    padded[:,N:] = data[:,::-1]

    if fft==numpy.fft.fft:
        fft_data = fft(padded,axis=-1)
    else:
        fft_data = fft(padded,axis=-1,threads=threads)

    freqs,wavelets = prepare_wavelet_fft_basis(fa,fb,nfreqs,N*2,w,float(Fs))

    if fft==numpy.fft.fft:
        result   = np.array([ifft(fft_data*wl,axis=1)[:,:N] for wl in wavelets])
    else:
        result   = np.array([ifft(fft_data*wl,axis=1,threads=threads)[:,:N] for wl in wavelets])

    return freqs,np.transpose(result,(1,0,2))


def fft_cwt_transposed_logspaced(data,fa,fb,w=4.0,nfreqs=None,threads=1,Fs=1000):
    '''
    Parameters
    ----------
    data: NCHANNELS × NTIMES np.array
    fa: positive float
        Low-frequency cutoff in Hz
    fb: positive float
        High-frequency cutoff in Hz
    
    Other Parameters
    ----------------
    w: positive float; default 4.0
        Wavelet base frequency; 
        Controls the time-frequency tradeoff
    nfreqs: positive float; default 0.1
        Frequency sampling nfreqs
    threads: positive int; default 1
        Number of CPU threads to use
    Fs: positive int; default 1000
        Sample rate
        
    
    Returns
    -------
    freqs: 
        Frequencies of each band, in Hz
    result: NCHANNELS × NFREQS × NTIMES np.array
        wavelet transform
    '''
    
    if nfreqs is None: nfreqs = int(round(fb-fa))
    if len(np.shape(data))==1:
        data = np.renp.shape(data,(1,)+np.shape(data))
    NCH,N = np.shape(data)
    if NCH>N:
        warn('MORE CHANNELS THAN DATA CHECK FOR TRANSPOSED')
        data = data.T
        N,NCH=NCH,N
    padded       = np.zeros((NCH,N*2),dtype=np.complex64)
    padded[:,:N] = data
    padded[:,N:] = data[:,::-1]
    
    if fft==numpy.fft.fft:
        fft_data = fft(padded,axis=-1)
    else:
        fft_data = fft(padded,axis=-1,threads=threads)
    freqs,wavelets = prepare_wavelet_fft_basis_logspace(fa,fb,nfreqs,N*2,w,Fs)
    
    if fft==numpy.fft.fft:
        result   = np.array([ifft(fft_data*wl,axis=1)[:,:N] for wl in wavelets])
    else:
        result   = np.array([ifft(fft_data*wl,axis=1,threads=threads)[:,:N] for wl in wavelets])
    return freqs,np.transpose(result,(1,0,2))




def _population_eigencoherence(lfp,fa,fb,w=4.0,nfreqs=0.1,Fs=1000):
    '''
    not implemented.
    
    Uses the eigenvalue spectrum of the pairwise coherence matrix.
    In the case of wavelets, each time-frequency point has one
    complex value.

    The matrix we build: $|z_i z_j|$

    See ramirez et al
    A GENERALIZATION OF THE MAGNITUDE SQUARED COHERENCE SPECTRUM FOR
    MORE THAN TWO SIGNALS: DEFINITION, PROPERTIES AND ESTIMATION

    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    raise NotImplementedError('This function was never implemented')
    

def _mtm_cohere():
    '''
    Multitaper coherence

    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    raise NotImplementedError('This function was never implemented')



############################################################################
############################################################################
if __name__=='__main__':
    #wavelet power test
    signal = randn(1000)
    Fs=1000
    for freq in arange(5,500,5):
        ws   = arange(4,30)
        M    = 2.*1.*ws*Fs/float(freq)
        clf()
        x = []
        bw = []
        for m,w in zip(M,ws):
            wl = normalized_morlet(m,w)
            a,b = fftfreq(len(wl),1./Fs),abs(fft(wl))
            #plot(a,b)
            df = a[1]-a[0]
            bw.append(sum(b)*df)
            s = convolve(signal,wl,'same')
            #print(m,w,mean(abs(s)),mean(abs(s)**2))
            x.append(var(s))

        bw = arr(bw)
        x = arr(x)
        plot(x/bw)
        positivey()
        print(freq,1/mean(bw/x))
