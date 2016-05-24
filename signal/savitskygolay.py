
from neurotools.tools import *
import numpy as np

def SGOrd(m,fc,fs):
    '''
    Fc = (N+1)/(3.2M-4.6)
    For fixed M, Fc
    N = Fc*(3.2M-4.6)-1
    '''
    fc = fc/(0.5*fs)
    return int(round(fc*(3.2*m-4.6)-1))

def SGKern(m,n):
    x = arange(-m,m+1)
    y = zeros(shape(x))
    y[m]=1
    k=poly1d(polyfit(x,y,n))(x)
    return k
    
def SGKernV(m,n):
    x = arange(-m,m+1)
    y = zeros(shape(x))
    y[m-1]=.5
    y[m+1]=-.5
    k=poly1d(polyfit(x,y,n))(x)
    return k

def SGKernA(m,n):
    x = arange(-m,m+1)
    y = zeros(shape(x))
    y[m-2]=.25
    y[m]  =-.5
    y[m+2]=.25
    k=poly1d(polyfit(x,y,n))(x)
    return k

def SGKernJ(m,n):
    x = arange(-m,m+1)
    y = zeros(shape(x))
    y[m-3]=.125
    y[m-1]=-.375
    y[m+1]=.375
    y[m+3]=-.125
    k=poly1d(polyfit(x,y,n))(x)
    return k

def SGfilt(m,fc,fs):
    n = SGOrd(m,fc,fs)
    return SGKern(m,n)

def SGfiltV(m,fc,fs):
    n = SGOrd(m,fc,fs)
    return SGKernV(m,n)

def SGfiltA(m,fc,fs):
    n = SGOrd(m,fc,fs)
    return SGKernA(m,n)

def SGfiltJ(m,fc,fs):
    n = SGOrd(m,fc,fs)
    return SGKernJ(m,n)

def SGaccelerate(x,m,fc,fs):
    n = len(x)
    x = concatenate([x[::-1],x,x[::-1]])
    x = np.convolve(x,SGfiltA(m,fc,fs),mode='same')
    x = x[n:n*2]
    return x*fs*fs

def SGjerk(x,m,fc,fs):
    n = len(x)
    x = concatenate([x[::-1],x,x[::-1]])
    x = np.convolve(x,SGfiltA(m,fc,fs),mode='same')
    x = x[n:n*2]
    return x*fs*fs*fs

def SGdifferentiate(x,m,fc,fs):
    n = len(x)
    x = concatenate([x[::-1],x,x[::-1]])
    x = np.convolve(x,SGfiltV(m,fc,fs),mode='same')
    x = x[n:n*2]
    return x*fs

def SGsmooth(x,m,fc,fs):
    n = len(x)
    x = concatenate([x[::-1],x,x[::-1]])
    x = np.convolve(x,SGfilt(m,fc,fs),mode='same')
    x = x[n:n*2]
    return x







