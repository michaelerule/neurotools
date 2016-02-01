"""
We are going to need to get the spike times, then open up the NS5 and
extract those points exactly. 

Thankfully it's all NS5 this time, so we know that the sample rate is
30 kilosamples so we can convert spike times to times in the neural data.

We will need, for each unit:

the session, area, channel number (IN RAW DATA)
and a list of spike times converted to frame number.

We will need to load the corresponding NS5 channel, 
High pass filter at 250Hz with zero phase 4th order butterworth, 

and grab the times surrounding the spikes.Currently we grabe 1.6 ms or
1600 microseconds which is 48 samples. I am .. hoping that the spike times
are set at the beginning of the sample window but we have no guarantees.

So lets do 3ms on either side of the spike. That's 90 samples before and
after the putatitve spike time.

We need to collect all these snippits in a nice matrix and store that to 
disk

These are the relevant NS5 files in /ldisk_2/mrule/archive
SPIKE/SPK120924/PMv/original/SPK120924_PMv_TT_FRG.ns5
SPIKE/SPK120924/M1_PMd/original/SPK120924_MI_PMd_TT_FRG001.ns5
SPIKE/SPK120918/PMv/original/SPK120918_PMv_TT_FRG002.ns5
SPIKE/SPK120918/M1_PMd/original/SPK120918_MI_PMd_TT_FRG002.ns5
SPIKE/SPK120925/PMv/original/SPK120925_PMv_TT_FRG001.ns5
SPIKE/SPK120925/M1_PMd/original/SPK120925_MI_PMd_TT_FRG001.ns5
RUSTY/RUS120521/PMv/original/RUSRH120521_PMv_TT_KG_TC_FRG001.ns5
RUSTY/RUS120521/M1_PMd/original/RUSRH120521_MI_PMd_TT_KG_TC_FRG001.ns5
RUSTY/RUS120518/PMv/original/RUSRH120518_PMv_TT_KG_TC_FRG001.ns5
RUSTY/RUS120518/M1_PMd/original/RUSRH120518_MI_PMd_TT_KG_TC_FRG001.ns5
RUSTY/RUS120523/PMv/original/RUSRH120523_PMv_TT_KG_TC_FRG001.ns5
RUSTY/RUS120523/M1_PMd/original/RUSRH120523_MI_PMd_TT_KG_TC_FRG001.ns5
"""

from matplotlib.colors import *
from pylab       import *
from scipy.io    import *
from collections import *
from itertools   import *
import sys
import os
import os, sys
from scipy.stats import describe
from pylab import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy import signal
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import * #.hsv_to_rgb(hsv)
import os, sys
from scipy.io import loadmat,savemat
from itertools import *
import pickle
from scipy.stats import *
from multiprocessing import Process, Pipe, cpu_count, Pool
from itertools import izip, chain
from pylab import *

def GMM(PCA,NCLASS=2):
    '''
    Gaussian mixture model
    # PDF = Pr(G) (2pi)^(k/2) |S|^(-1/2) exp[-1/2 (x-mu)' S^(-1) (x-mu) ]
    # logPDF = logPr(G) k/2 log(2pi) - 1/2log(|S|)-1/2(x-mu)'S^(-1)(x-mu)
    # Pr inverse monotonic with logPr(G) - log(|S|) - (x-mu)' S^(-1) (x-mu)
    '''
    N        = shape(PCA)[1]
    initsize = N/NCLASS
    classes  = zeros((N,))
    oldclasses  = zeros((N,))
    Pr       = zeros((N,NCLASS))
    partition = (2*pi)**(-0.5*NCLASS)
    for i in range(NCLASS):
        classes[i*initsize:(i+1)*initsize] = i
    for ii in range(2000):
        for i in range(NCLASS):
            c = PCA[:,classes==i]
            Mu = mean(c,1)
            Cm = cov((c.T-Mu).T) 
            k  = shape(c)[1]
            Pm = pinv(Cm)
            center = (PCA.T-Mu)
            normalize = partition*k/(N+1.)/sqrt(det(Cm))
            Pr[:,i] = exp(-0.5*array([dot(x,dot(Pm,x.T)) for x in center]))*normalize
        oldclasses[:]=classes
        classes = argmax(Pr,1)
        if all(oldclasses==classes):break
    classification = (Pr.T/sum(Pr,1)).T
    return classification,Pr

def GMM1D(PCA,NCLASS=2):
    '''
    1-D Gaussian mixture model
    # PDF = Pr(G) (2pi)^(k/2) |S|^(-1/2) exp[-1/2 (x-mu)' S^(-1) (x-mu) ]
    # logPDF = logPr(G) k/2 log(2pi) - 1/2log(|S|)-1/2(x-mu)'S^(-1)(x-mu)
    # Pr inverse monotonic with logPr(G) - log(|S|) - (x-mu)' S^(-1) (x-mu)
    '''
    PCA      = squeeze(PCA)
    N        = len(PCA)
    initsize = N/NCLASS
    classes  = zeros((N,))
    Pr       = zeros((N,NCLASS))
    partition = (2*pi)**(-0.5*NCLASS)
    for i in range(NCLASS):
        classes[i*initsize:(i+1)*initsize] = i
    for ii in range(2000):
        for i in range(NCLASS):
            c  = PCA[classes==i]
            Mu = mean(c)
            Cm = var(c)
            k  = len(c)
            Pm = 1./Cm
            center = (PCA-Mu)
            normalize = partition*k/(N+1.)/sqrt(Cm)
            Pr[:,i] = exp(-0.5*Pm*center**2)*normalize
        classes = argmax(Pr,1)
    classification = (Pr.T/sum(Pr,1)).T
    return classification,Pr

def realign(snip):
    '''
    Realign waveforms to peak
    '''
    i = argmin(snip)
    n = len(snip)
    m = n/2
    shiftback = i-m
    result = zeros(shape(snip))
    if shiftback==0:  result=snip
    elif shiftback>0: result[0:-shiftback]=snip[shiftback:]
    else:             result[-shiftback:] =snip[0:shiftback]
    return result

def realign_special(snip):
    '''
    Realign waveforms to peak, pad out missing values
    '''
    #expect length 240
    #min should be set at 87
    i = argmin(snip)
    n = len(snip)
    assert n==240
    m = 87
    shiftback = i-m
    result = zeros(shape(snip))
    if shiftback==0:  result=snip
    elif shiftback>0: 
        result[0:-shiftback]=snip[shiftback:]
        result[-shiftback:] =snip[-1]
    else:             
        result[-shiftback:] =snip[0:shiftback]
        result[:-shiftback] =snip[0]
    return result
    
def getFWHM(wf):
    '''
    Full width half maximum
    '''
    m = np.min(wf)
    x = 0.0# np.max(wf)
    h = (m+x)/2.
    ok = int32(wf<=h)
    start = find(diff(ok)==1)
    stop  = find(diff(ok)==-1)
    if len(start)!=1: return NaN
    if len(stop) !=1: return NaN
    start = start[0]
    stop  = stop[0]
    if start>=stop: return NaN
    return stop-start

def getPVT(wf):
    '''
    peak to valley time
    '''
    a = argmin(wf)
    return argmax(wf[a:])

def getWAHP(wf):
    '''
    Width at half peak
    '''
    x     = np.max(wf[argmin(wf):])
    h     = x*0.5
    m     = np.argmin(wf)
    ok    = int32(wf>=h)
    edge  = diff(ok)
    start = find(edge==1)
    stop  = find(edge==-1)
    start = [s for s in start if s>m]
    stop  = [s for s in stop  if s>m]
    if len(start)!=1: return NaN
    if len(stop) !=1: return NaN
    a = start[0]
    b = stop[0]
    if b<=a: return NaN
    return b-a

def getPT(wf):
    '''
    Peak-trough duration
    '''
    m  = argmin(wf)
    wf = wf[m::-1]
    k  = argmax(wf)
    return k

def getPTHW(wf):
    m  = argmin(wf)
    wf = wf[m::-1]
    h  = 0.5*max(wf)
    ok    = int32(wf>=h)
    edge  = diff(ok)
    start = find(edge==1)
    stop  = find(edge==-1)
    if len(start)==0: return NaN
    if len(stop) ==0: return NaN
    a = start[0]
    b = stop[0]
    if b<=a: return NaN
    return b-a

def getPHP(wf):
    m  = argmin(wf)
    x  = np.min(wf)
    wf = wf[m::-1]
    h  = max(wf)
    return h/x

def upsample(wf,FACTOR=4):
    N=len(wf)
    if N%2==0:
        fftcoeff = fft(wf)
        newcoeff = complex64(zeros(N*4))
        newcoeff[1:N/2+1]      = fftcoeff[1:N/2+1]
        newcoeff[-1:-N/2-1:-1] = conj(fftcoeff[1:N/2+1])
        return real(ifft(newcoeff)*4)
    else:
        fftcoeff = fft(wf)
        newcoeff = complex64(zeros(N*4))
        newcoeff[1:N/2+1]      = fftcoeff[1:N/2+1]
        newcoeff[-1:-N/2:-1] = conj(fftcoeff[1:N/2+1])
        return real(ifft(newcoeff)*4)

def process((i,f)):
    sys.stderr.write('\r'+'\t'*8+f+' loading..')
    sys.stderr.flush()
    data = loadmat('./extracted_ns5_spikes_nohighpass/'+f)
    sys.stderr.write('\r'+'\t'*8+f+' aligning..')
    sys.stderr.flush()
    s=data['snippits']
    s=((s.T-mean(s,1))/std(s,1)).T
    wf = mean(s,0)
    sys.stderr.write('\r'+'\t'*8+f+' computing..')
    sys.stderr.flush()
    z = array(map(upsample,s))
    z = z[:,80*4:140*4]
    z = array(map(realign_special,z))
    mwf = nanmean(z,0)
    # we need to upsample and operate over the averaged waveform
    ahpw = getWAHP(mwf)/4.0
    pvt  = getPVT (mwf)/4.0
    fwhm = getFWHM(mwf)/4.0
    pt   = getPT  (mwf)/4.0
    pthw = getPTHW(mwf)/4.0
    php  = getPHP (mwf)
    return i,f,wf,ahpw,pvt,fwhm,pt,pthw,php,mwf





