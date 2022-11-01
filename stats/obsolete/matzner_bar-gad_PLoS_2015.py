#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
In a paper in PLoS computational biology, April 2015, titled "Quantifying Spike
Train Oscillations: Biases, Distortions and Solutions", Ayala Matzner and Izhar 
Bar-Gad solve explicitly for the power spectrum of a point process with an
underlying rate function that is the sum of a sinusoidal and constant term. 
They use this to quantify the spectral content of the point process in a way
that is not biased by firing rate or the problematic power spectral properties
of point processes.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

'''
Matzner and Bar-Gad use the model
lambda(t) = Ro ( 1 + m cos(2pi f t) )

A log-linear model might be more natural
ln(lambda(t)) = Ro + m cos(2pi f t + phase)
We can explore that later, let's reproduce Matzner and Bar-Gad for now

Matzner and Bar-Gad define an estimate of the "modulation index" parameter "m"
m = abs(
    2 sqrt( S - r ) / ( r sqrt(T) )
    )
    
Based on an approximation of the power at the modulated frequency
S = r ( 1+ r T m^2 / 4 )

This approache can generate imaginary power estimates when the estimated
mean rate is higher than the power. Can we re-do it in log-power? 
'''

if __name__=='__main__':
    from os.path import *
    from multitapertools import *
    from numpy import *
    from matplotlib.pyplot import clf,plot
    from numpy.random import rand

    # Simulation
    Fs = 1000 #Hz
    T  = 10   #seconds
    w  = 20   #Hz
    r  = 55   #Hz
    m  = 0.5  #"modulation index"

    t = arange(Fs*T,dtype=float32)/Fs

    NTAPER = 5

    clf()
    for r in linspace(10,45,5):
        rate = r * (1 + m* cos(2*pi*w*t))
        spikes = int32(rand(Fs*T)<(rate/Fs))
        ff,pp = multitaper_spectrum(spikes,NTAPER,Fs=1000.0)
        rHat = mean(spikes)
        mHat = abs( 2* sqrt( pp**2  - rHat ) / ( rHat * sqrt(T) ) )
        plot(ff,mHat)


    from scipy.signal import butter, filtfilt, lfilter
    def bandfilter(data,fa=None,fb=None,Fs=1000.,order=4,zerophase=True,bandstop=False):
        N = len(data)
        assert len(shape(data))==1
        padded = zeros(2*N,dtype=data.dtype)
        padded[N/2:N/2+N]=data
        padded[:N/2]=data[N/2:0:-1]
        padded[N/2+N:]=data[-1:N/2-1:-1]
        if not fa==None and not fb==None:
            if bandstop:
                b,a  = butter(order,array([fa,fb])/(0.5*Fs),btype='bandstop')
            else:
                b,a  = butter(order,array([fa,fb])/(0.5*Fs),btype='bandpass')
        elif not fa==None:
            # high pass
            b,a  = butter(order,fa/(0.5*Fs),btype='high')
            assert not bandstop
        elif not fb==None:
            # low pass
            b,a  = butter(order,fb/(0.5*Fs),btype='low')
            assert not bandstop
        else:
            assert 0
        if zerophase:
            return filtfilt(b,a,padded)[N/2:N/2+N]
        else:
            return lfilter(b,a,padded)[N/2:N/2+N]
        assert 0
            
    # now try it with a broad spectrum
    m = 15
    NTAPER = 20
    T = 10
    clf()
    for r in linspace(10,45,5):
        noise = randn(Fs*T)
        bandpass = bandfilter(noise,10,45)
        rate   = r * (1 + m* bandpass)#cos(2*pi*w*t))
        spikes = int32(rand(Fs*T)<(rate/Fs))
        ff,pp  = multitaper_spectrum(spikes,NTAPER,Fs=1000.0)
        rHat   = mean(spikes)
        mHat   = abs( 2* sqrt( pp**2 - rHat ) / ( rHat * sqrt(T) ) )
        plot(ff,mHat)

