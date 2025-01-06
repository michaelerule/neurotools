#!/usr/bin/python3
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
try:
  from scipy.signal import morlet
except:
  print('neurotools.signal.morlet: migrating to pywavelets is TODO')
import warnings

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
    freqs = np.linspace(fa,fb,nfreqs)
    M     = np.int32(np.round(2.*1.*w*Fs/freqs))
    # we have to be careful about padding. In general we want the wavelets
    # to all be centered in the array before we take the FFT, so the padding
    # should be symmetric
    #
    # Modification: now normalizing wavelet amplitude to 1
    allwl = []
    fftwl = []
    for i,m in enumerate(M):
        wl = morlet(m,w=w)
        wl = wl/sum(abs(wl))
        N = len(wl)
        if N>L:
            chop_begin = (N-L)//2
            chop_end   = (N-L)-chop_begin
            wl = wl[chop_begin:-chop_end]
            N = len(wl)
            assert N==L
        padded = np.zeros(L,dtype=np.complex64)
        start  = (L-N)//2
        padded[start:start+N]=wl
        reordered = np.zeros(L,dtype=np.complex64)
        reordered[L//2:]=padded[:L//2]
        reordered[:L//2]=padded[L//2:]
        allwl.append(reordered)
        fftwl.append(fft(reordered))
    return freqs,np.array(fftwl)

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
    
def geometric_window(c,w):
    '''
    Gemoetrically centered frequency window.
    
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
    if not c>0: raise ValueError(
        'The center of the window should be positive')
    if not w>=0:raise ValueError(
        'The window size should be non-negative')
    lgwindow = (w+np.sqrt(w**2+4*c**2))/(2*c)
    fa       = c/lgwindow
    fb       = c*lgwindow
    return fa,fb
    
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
    data,
    fa,
    fb,
    w=4.0,
    nfreqs=1000,
    Fs=1000.0,
    threads=1):
    '''
    Compute morelet spectrogam of a list of time-domain
    signals (possibly in parallel if you have a threaded
    installation of the FFTW library available). 
    
    This version spaces ``nfreqs`` frquencies uniformly
    between ``fa`` and ``fb``, inclusive. For logarithmic
    spacing, see ``fft_cwt_transposed_logspaced()``.
    
    Parameters
    ----------
    data : numeric
        NCH x Ntimes list of signals to transform
    fa: float
        Lowest frequency to extract
    fb: float
        Highest frequency to extract
        
    Other Parameters
    ----------------
    4: float; default 4.0
        Time/frequency morlet width parameter
    nfreqs: int; default 1000
        Number of frequency components to use
    Fs: float; default 1000.0
        Sample rate in Hz
    threads: int; default 1
        Number of threads to use
    
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
        warnings.warn('MORE CHANNELS THAN DATA, CHECK IF TRANSPOSED')
    padded       = np.zeros((NCH,N*2),dtype=np.complex64)
    padded[:,:N] = data
    padded[:,N:] = data[:,::-1]

    if fft==numpy.fft.fft:
        fft_data = fft(padded,axis=-1)
    else:
        fft_data = fft(padded,axis=-1,threads=threads)

    freqs,wavelets = prepare_wavelet_fft_basis(fa,fb,nfreqs,N*2,w,float(Fs))

    if fft==numpy.fft.fft:
        result = np.array([
            ifft(fft_data*wl,axis=1)[:,:N] 
            for wl in wavelets])
    else:
        result = np.array([
            ifft(fft_data*wl,axis=1,threads=threads)[:,:N] 
            for wl in wavelets])

    return freqs,np.transpose(result,(1,0,2))


def fft_cwt_transposed_logspaced(
    data,
    fa,
    fb,
    w=4.0,
    nfreqs=None,
    Fs=1000.0,
    threads=1,
    ):
    '''
    Compute morelet spectrogam of a list of time-domain
    signals (possibly in parallel if you have a threaded
    installation of the FFTW library available). 
    
    This version spaces ``nfreqs`` frquencies logarithmically
    between ``fa`` and ``fb``, inclusive. For uniform
    spacing, see ``fft_cwt_transposed()``.
    
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
    Fs: positive int; default 1000
        Sample rate
    threads: positive int; default 1
        Number of CPU threads to use
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
# New routines 2023 03 27 mer

def convenient_morlet(
    N,        # Number of samples
    f0,       # Morlet frequency Hz
    sigma_ms, # Standard deviation in ms
    Fs,
    ):
    sigma = sigma_ms * Fs/1000 # Standard deviation in bins
    scale = N/(sigma*4*np.pi)     # Morlet scale parameter
    width = N*f0/2/scale/Fs    # Morlet width parameter
    wl    = scipy.signal.morlet(N,width,scale)
    return wl/np.sum(np.abs(wl)**2)**0.5

def get_gentle_morlets_ft(
    N,
    flo = 2,  # Hz
    f0  = 15, # Hz 
    fhi = 45, # Hz
    sigma_lo = 1000/6, # Lowest temporal resolution in ms
    sigma_md = 100,    # target temporal resoltion at f0
    nfreqs   = 100,    # Number of frequency bins (linearly spaced)
    Fs       = 1000,   # Sample rate
    ):
    ff = np.linspace(flo,fhi,nfreqs)
    # Adjust bandwidth gradually
    k = np.log(sigma_md/sigma_lo) / np.log(f0/flo)
    sigmas = np.exp(np.log(sigma_lo) + k*np.log(ff/flo))
    wl  = np.complex64([convenient_morlet(N, f, sms, Fs = Fs) for f,sms in zip(ff,sigmas)])
    wl  = scipy.fft.fftshift(wl,axes=1)
    wft = scipy.fft.fft(wl,axis=1)
    return ff,wft

def gentle_morlet_psd(
    u, # Signal to process
    flo = 2,  # Hz
    f0  = 15, # Hz 
    fhi = 45, # Hz
    sigma_lo = 1000/6, # Lowest temporal resolution in ms
    sigma_md = 100,    # target temporal resoltion at f0
    nfreqs   = 100,    # Number of frequency bins (linearly spaced)
    Fs       = 1000,   # Sample rate
    ):
    '''
    Returns
    -------
    psd: np.ndarray
        ``NFREQS x NSAMPLES`` power spectrum
    '''
    assert flo<f0<fhi
    assert sigma_md < sigma_lo
    
    # Remove trend and add padding
    N0 = len(u)
    ii = np.arange(N0)
    m,b = np.polyfit(ii,u,1)
    u0 = m*ii + b
    u -= u0
    Nr = N0//2
    u = np.concatenate([u,u[::-1]])
    u = np.roll(u,Nr)
    
    N  = len(u)
    
    ff,wft = get_gentle_morlets_ft(N,flo,f0,fhi,sigma_lo,sigma_md,nfreqs,Fs)
    uft = scipy.fft.fft(u)
    mwt = scipy.fft.ifft(uft*wft,axis=1)
    psd = abs(mwt)**2
    
    # Remove padding
    psd = np.roll(psd,-Nr,axis=1)
    psd = psd[:,:N0]
    return ff,psd
