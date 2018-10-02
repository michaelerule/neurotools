#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function        

def virtual_reference_line_noise_removal(lfps):
    '''
    Accepts an array of LFP recordings (first dimension should be 
    channel number, second dimension time ).
    Sample rate assumed 1000Hz
    
    Extracts the mean signal within 2.5 Hz of 60Hz.
    For each channel, removes the projection of the LFP signal onto this
    estimated line noise signal.
    
    Note: for some reason this doesn't always work that well.
    '''
    hbw=5
    filtered = [bandfilter(x,60-hbw,60+hbw) for x in lfps]
    noise    = mean(filtered,0)
    scale    = 1./dot(noise,noise)
    removed  = [x-dot(x,noise)*scale*noise for x in lfps]
    return removed

def band_stop_line_noise_removal(lfps):
    '''
    removes line noise using band-stop at 60Hz and overtones
    '''
    hbw   = 10
    freqs = [60,120,180,240,300]
    lfps = array(lfps)
    for i,x in enumerate(lfps):
        for f in freqs:
            lfps[i,:] = bandfilter(lfps[i,:],f-hbw,f+hbw,bandstop=1)
    return lfps
    

