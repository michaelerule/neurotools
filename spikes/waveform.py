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

"""TODO: title here

TODO: module description here
"""

import os, sys, pickle
from scipy import signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
from neurotools.signal.signal import upsample

'''
TODO: update code to make these specific imports
( as-is this is breaking sphinx autodoc / apidoc )
'''
'''
from collections import *
from itertools   import *
from pylab       import *
from scipy.io    import *
from scipy.stats import *
from   matplotlib.colors import *
from neurotools.stats.gmm import *
'''

def realign(snip):
    '''
    Realign waveforms to peak
    '''
    i = np.argmin(snip)
    n = len(snip)
    m = n/2
    shiftback = i-m
    result = np.zeros(snip.shape)
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
    i = np.argmin(snip)
    n = len(snip)
    assert n==240
    m = 87
    shiftback = i-m
    result = np.zeros(np.shape(snip))
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
    ok = np.int32(wf<=h)
    start = find(np.diff(ok)==1)
    stop  = find(np.diff(ok)==-1)
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
    a = np.argmin(wf)
    return np.argmax(wf[a:])

def getWAHP(wf):
    '''
    Width at half peak
    '''
    x     = np.max(wf[np.argmin(wf):])
    h     = x*0.5
    m     = np.argmin(wf)
    ok    = np.int32(wf>=h)
    edge  = np.diff(ok)
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
    m  = np.argmin(wf)
    wf = wf[m::-1]
    k  = np.argmax(wf)
    return k

def getPTHW(wf):
    m  = np.argmin(wf)
    wf = wf[m::-1]
    h  = 0.5*max(wf)
    ok    = np.int32(wf>=h)
    edge  = np.diff(ok)
    start = np.find(edge==1)
    stop  = np.find(edge==-1)
    if len(start)==0: return NaN
    if len(stop) ==0: return NaN
    a = start[0]
    b = stop[0]
    if b<=a: return NaN
    return b-a

def getPHP(wf):
    m  = np.argmin(wf)
    x  = np.min(wf)
    wf = wf[m::-1]
    h  = max(wf)
    return h/x


def minmax(wf):
    ''' 
    Normalize waveform based on extrema 
    '''
    a = np.min(wf)
    b = np.max(wf)
    wf -= a
    wf /= (b-a)
    return wf

def normalized_waveform(wf):
    wf = upsample(zscore(wf),5)
    wf = realign_special(wf)
    wf = (wf-np.mean(wf[40:200]))/np.std(wf[40:200])
    return wf

def is_thin(wf,thr=0.98,time=123):
    '''
    Determine whether a mean-waveform is a thin spike.
    Uses amplituce 300 μs post-spike.
    This procedure was trained on well-isolated cells.
    See the 20160802_waveform_segmentation notebook for how
    threshold was derived
    '''
    wf = normalized_waveform(wf)
    a300 = wf[time]
    return a300>thr

def is_thin_pvt(wf,thr=52.0349055393):
    '''
    Determine whether a mean-waveform is a thin spike.
    Uses peak-to-valley time
    '''
    pvt = getPVT(normalized_waveform(wf))
    return pvt<thr

def process(i_f):
    '''
    Get high-dimensional feature description of data.
    TODO: remove; how did this even get here?
    '''
    '''
    (i,f) = i_f
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
    '''
    pass




