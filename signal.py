
from neurotools.tools import *
from matplotlib.mlab import *
from neurotools.getfftw import *
from scipy.signal.signaltools import fftconvolve,hilbert
from numpy.random import *

def gaussian_kernel(sigma):
    '''
    generate Guassian kernel for smoothing
    sigma: standard deviation, >0
    '''
    assert sigma>0
    K = ceil(sigma)
    N = K*2+1
    K = exp( - (arange(N)-K)**2 / (2*sigma**2) )
    K *= 1./sum(K)
    return K

def gaussian_smooth(x,sigma):
    '''
    Smooth signal x with gaussian of standard deviation sigma
    
    sigma: standard deviation
    x: 1D array-like signal
    '''
    K = gaussian_kernel(sigma)
    return convolve(x,K,'same')

def zscore(x):
    ss = std(x,0)+1e-30
    #if std(x,0)<1e-60: return x
    return (x-mean(x,0))/ss

def local_maxima(x):
   '''
   returns signal index and values at those indecies
   '''
   t = find(diff(sign(diff(x)))<0)+1
   return t,x[t]

def amp(x):
    return abs(hilbert(x))
    
def getsnips(signal,times,window):
    times = times[times>window]
    times = times[times<len(signal)-window-1]
    snips = array([signal[t-window:t+window+1] for t in times])
    return snips

def triggeredaverage(signal,times,window):
    return mean(getsnips(signal,times,window),0)

def gettriggeredstats(signal,times,window):
    s = getsnips(signal,times,window)
    return mean(s,0),std(s,0),std(s,0)/sqrt(len(times))*1.96

def padout(data):
    N = len(data)
    assert len(shape(data))==1
    padded = zeros(2*N,dtype=data.dtype)
    padded[N/2:N/2+N]=data
    padded[:N/2]=data[N/2:0:-1]
    padded[N/2+N:]=data[-1:N/2-1:-1]
    return padded

def padin(data):
    N = len(data)
    assert len(shape(data))==1
    return data[N/2:N/2+N]

from scipy.signal import butter, filtfilt, lfilter
def bandfilter(data,fa=None,fb=None,Fs=1000.,order=4,zerophase=True,bandstop=False):
    '''
    IF fa is None, assumes high pass with cutoff fb
    IF fb is None, assume low pass with cutoff fa
    Array can be any dimension, filtering performed over last dimension
    '''
    N = shape(data)[-1]
    padded = zeros(shape(data)[:-1]+(2*N,),dtype=data.dtype)
    padded[...,N/2  :N/2+N] = data
    padded[...,     :N/2  ] = data[...,N/2:0    :-1]
    padded[...,N/2+N:     ] = data[...,-1 :N/2-1:-1]
    if not fa==None and not fb==None:
        if bandstop:
            b,a = butter(order,array([fa,fb])/(0.5*Fs),btype='bandstop')
        else:
            b,a = butter(order,array([fa,fb])/(0.5*Fs),btype='bandpass')
    elif not fa==None:
        # high pass
        b,a  = butter(order,fa/(0.5*Fs),btype='high')
        assert not bandstop
    elif not fb==None:
        # low pass
        b,a  = butter(order,fb/(0.5*Fs),btype='low')
        assert not bandstop
    else: raise Exception('Both fa and fb appear to be None')
    return (filtfilt if zerophase else lfilter)(b,a,padded)[...,N/2:N/2+N]
    assert 0

def box_filter(data,smoothat):
    '''
    Smooths data by convolving with a size smoothat box
    provide smoothat in units of frames i.e. samples (not ms or seconds)
    '''
    N = len(data)
    assert len(shape(data))==1
    padded = zeros(2*N,dtype=data.dtype)
    padded[N/2:N/2+N]=data
    padded[:N/2]=data[N/2:0:-1]
    padded[N/2+N:]=data[-1:N/2-1:-1]
    smoothed = fftconvolve(padded,ones(smoothat)/float(smoothat),'same')
    return smoothed[N/2:N/2+N]

def median_filter(x,window=100,mode='same'):
    '''
    median_filter(x,window=100,mode='same')
    Filters a signal by calculating the median in a sliding window of
    width 'window'
    
    mode='same' will compute median even at the edges, where a full window
        is not available
    
    mode='valid' will compute median only at points where the full window
        is available
    
    '''
    n = shape(x)[0]
    if mode=='valid':
        filtered = [median(x[i:i+window]) for i in range(n-window)]
        return array(filtered)
    if mode=='same':
        warn('EDGE VALUES WILL BE BAD')
        w = window / 2
        filtered = [median(x[max(0,i-w):min(n,i+w)]) for i in range(n)]
        return array(filtered)
    assert 0
        
def rewrap(x):
    return (x+pi)%(2*pi)-pi

def pdiff(x):
    return rewrap(diff(x))

def pghilbert(x):
    return pdiff(angle(hilbert(x)))

def fudge_derivative(x):
    n = len(x)+1
    result = zeros(n)
    result[1:]   += x
    result[:-1]  += x
    result[1:-1] *= 0.5
    return result

def ifreq(x,Fs=1000,mode='pad'):
    pg = pghilbert(x)
    pg = pg/(2*pi)*Fs
    if mode=='valid':
        return pg # in Hz
    if mode=='pad':
        return fudgeDerivative(pg)
    assert 0
        
def zscore(x):
    return (x-mean(x,0))/std(x,0)

def unwrap(h):
    d = fudgeDerivative(pdiff(h))
    return cumsum(d)

def ang(x):
    return angle(hilbert(x))
    
def randband(N,fa=None,fb=None):
    return zscore(bandfilter(randn(N*2),fa=fa,fb=fb))[N/2:N/2+N]
    
def arenear(b,K=5):
    for i in range(1,K+1):
        b[i:] |= b[:-i]
    for i in range(1,K+1):
        b[:-i] |= b[i:]
    return b

def aresafe(b,K=5):
    for i in range(1,K+1):
        b[i:] &= b[:-i]
    for i in range(1,K+1):
        b[:-i] &= b[i:]
    return b

def get_edges(signal):
    starts = list(find(diff(int32(signal))==1))
    stops  = list(find(diff(int32(signal))==-1))
    if signal[0]: starts = [0]+starts
    if signal[-1]: stops = stops + [len(signal)]
    return starts, stops 
    
def set_edges(edges,N):
    '''
    Converts list of start, stop times over time period N into a [0,1]
    array which is 1 for any time between a [start,stop)
    edge info outsize [0,N] results in undefined behavior
    '''
    x = zeros(shape=(N,),dtype=np.int32)
    for (a,b) in edges:
        x[a:b]=1
    return x

def phase_rotate(s,f,Fs=1000.):
    '''
    Only the phase advancement portion of a resonator.
    See resonantDrive
    '''
    theta = f*2*pi/Fs
    s *= exp(1j*theta)
    return s

def fm_mod(freq):
    N = len(freq)
    signal = [1]
    for i in range(N):
        signal.append(phaseRotate(signal[-1],freq[i]))
    return array(signal)[1:]

def pieces(x,thr=4):
    dd = diff(x)
    br = [0]+list(find(abs(dd)>thr))+[len(x)]
    ps = []
    for i in range(len(br)-1):
        a,b = br[i:][:2]
        a+=1
        if a==b: continue
        ps.append((range(a,b),x[a:b]))
    return ps

def median_block(data,N=100):
    '''
    blocks data by median over last axis
    '''
    N = int(N)
    L = shape(data)[-1]
    B = L/N
    D = B*N
    if D!=N: warn('DROPPING LAST BIT, NOT ENOUGH FOR A BLOCK')
    data = data[...,:D]
    data = reshape(data,shape(data)[:-1]+(B,N))
    return median(data,axis=-1)   
    
def mean_block(data,N=100):
    '''
    blocks data by median over last axis
    '''
    N = int(N)
    L = shape(data)[-1]
    B = L/N
    D = B*N
    if D!=N: warn('DROPPING LAST BIT, NOT ENOUGH FOR A BLOCK')
    data = data[...,:D]
    data = reshape(data,shape(data)[:-1]+(B,N))
    return mean(data,axis=-1)

zc = zscore

def phase_randomize(signal):
    N = len(signal)
    x = fft(signal)
    if N%2==0: # N is even
        rephase = exp(1j*2*pi*rand((N-2)/2))
        rephase = concatenate([rephase,[sign(rand()-0.5)],conj(rephase[::-1])])
    else: # N is odd
        rephase = exp(1j*2*pi*rand((N-1)/2))
        rephase = append(rephase,conj(rephase[::-1]))
    rephase = append([1],rephase)
    x *= rephase
    return ifft(x)

def phase_randomize_from_amplitudes(amplitudes):
    '''
    phase_randomize_from_amplitudes(amplitudes)
    treats input amplitudes as amplitudes of fourier components
    '''
    N = len(amplitudes)
    x = complex128(amplitudes) # need to make a copy
    if N%2==0: # N is even
        rephase = exp(1j*2*pi*rand((N-2)/2))
        rephase = concatenate([rephase,[sign(rand()-0.5)],conj(rephase[::-1])])
    else: # N is odd
        rephase = exp(1j*2*pi*rand((N-1)/2))
        rephase = append(rephase,conj(rephase[::-1]))
    rephase = append([1],rephase)
    x *= rephase
    return real(ifft(x))

def estimate_padding(fa,fb,Fs=1000):
    bandwidth  = fb if fa is None else fa if fb is None else min(fa,fb)
    wavelength = Fs/bandwidth
    padding    = int(ceil(2.5*wavelength))
    return padding

def phaserand(signal):
    assert 1==len(shape(signal))
    N = len(signal)
    if N%2==1:
        # signal length is odd. 
        # ft will have one DC component then symmetric frequency components
        randomize  = exp(1j*rand((N-1)/2))
        conjugates = conj(randomize)[::-1]
        randomize  = append(randomize,conjugates)
    else:
        # signal length is even 
        # will have one single value at the nyquist frequency
        # which will be real and can be sign flipped but not rotated
        flip = 1 if rand(1)<0.5 else -1
        randomize  = exp(1j*rand((N-2)/2))
        conjugates = conj(randomize)[::-1]
        randomize  = append(randomize,flip)
        randomize  = append(randomize,conjugates)
    # the DC component is not randomized
    randomize = append(1,randomize)
    # take FFT and apply phase randomization
    ff = fft(signal)*randomize
    # take inverse
    randomized = ifft(ff)
    return real(randomized)
    
def lowpass_filter(x, cut=10, Fs=1000, order=4):
    return bandfilter(x,fb=cut,Fs=fs,order=order)
    
def highpassFilter(x, cut=40, Fs=240, order=2):
    return bandfilter(x,fa=cut,Fs=Fs,order=order)

def fdiff(x,fs=240.):
    return (x[2:]-x[:-2])*fs*.5

def killSpikes(x):
    x = array(x)
    y = zscore(highpassFilter(x))
    x[y<-1] = nan
    x[y>1] = nan
    for s,e in zip(*get_edges(isnan(x))):
        a = x[s-1]
        b = x[e+1]
        print s,e,a,b
        x[s:e+2] = linspace(a,b,e-s+2)
    return x

def peak_within(freqs,spectrum,fa,fb):
    '''
    Find maximum within a band
    '''
    # clean up arguments
    order    = argsort(freqs)
    freqs    = array(freqs)[order]
    spectrum = array(spectrum)[order]
    start = find(freqs>=fa)[0]
    stop  = find(freqs<=fb)[-1]+1
    index = argmax(spectrum[start:stop]) + start
    return freqs[index], spectrum[index]


def zeromean(x):
    return x-mean(x)



