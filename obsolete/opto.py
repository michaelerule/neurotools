#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Routines specific to working with Optogenetics datasets from
the Donoghue lab. These routines have yet to be ported to make 
them general. This module refers to and depends on 
unpublished data. 

You'll need to set the `opto_dataset` path variable to the
location of 
`TOMMY_MI_121101_full_trial_continuous_square_pulse_6mW001.mat`.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import sys
__PYTHON_2__ = sys.version_info<(3, 0)


opto_dataset = '/home/mrule/Workspace2/Optogenetics_data/TOMMY_MI_121101_full_trial_continuous_square_pulse_6mW001';

import sys, os
sys.path.insert(1,os.path.expanduser('~/Dropbox/bin'))
import neurotools
from neurotools.tools import metaloadmat,globalize
import numpy as np

@memoize
def opto_get_events_passive(opto_dataset):
    start,stop = metaloadmat(opto_dataset)['events']
    return start, stop

@memoize
def opto_get_all_lfp_quick(opto_dataset):
    try:
        data = metaloadmat(opto_dataset+'_compact')['lfp']
    except:
        data = metaloadmat(opto_dataset)['lfp']
    return np.array(data)

@memoize
def opto_get_map(opto_dataset):
    try:
        data = metaloadmat(opto_dataset+'_compact')['arrayChannelMap']
    except:
        data = metaloadmat(opto_dataset)['arrayChannelMap']
    return np.array(data)

@memoize
def opto_get_laser(opto_dataset):
    return np.array(metaloadmat(opto_dataset)['laser'][0])

@memoize
def opto_get_lfp(opto_dataset,channel):
    '''
    Retrieves channel or channels from opto LFP dataset
    Channels are 1-indexed

    Parameters
    ----------
    opto_dataset : string
        path or string identifier for a dataset
    channel:
        1-indexed channel ID or None to return a NTimes x NChannel array
        of all LFP data
    '''
    if channel is None:
        return metaloadmat(opto_dataset)['LFP'].T
    else:
        assert channel>=1
        return metaloadmat(opto_dataset)['LFP'][:,channel-1]

def __opto_get_lfp_filtered_helper__(params):
    (i,data,fa,fb,Fs,order) = params
    return i,bandfilter(data,fa,fb,Fs,order)

def opto_get_lfp_filtered(opto_dataset,channel,fa,fb,order=4):
    '''
    Retrieves channel or channels from opto LFP dataset
    Channels are 1-indexed

    Parameters
    ----------
    opto_dataset : string
        path or string identifier for a dataset
    channel:
        1-indexed channel ID or None to return a NTimes x NChannel array
        of all LFP data
    fa:
        low frequency of band-pass, or 'None' to use a low-pass filter.
        if fb is 'None' then this is the cutoff for a high-pass filter.
    fb:
        high-frequency of band-pass, or 'None to use a high-pass filter.
        if fa is 'None' then this is the cutoff for a low-pass filter
    '''
    Fs = opto_get_Fs(opto_dataset)
    if channel is None:
        data = metaloadmat(opto_dataset)['LFP'].T
        #return array([bandfilter(x,fa,fb,Fs,order) for x in data])
        problems = [(i,x,fa,fb,Fs,order) for i,x in enumerate(data)]
        data = array(parmap(__opto_get_lfp_filtered_helper__,problems))
        return squeeze(data)
    else:
        assert channel>=1
        data = metaloadmat(opto_dataset)['LFP'][:,channel-1]
        return bandfilter(data,fa,fb,Fs,order)

@memoize
def opto_get_all_lfp_analytic_quick(opto_dataset,fa,fb):
    Fs = 1000.0
    order = 4
    data = metaloadmat(opto_dataset+'_compact')['lfp']
    data = data.transpose((0,2,1))
    data = hilbert(bandfilter(data,fa,fb,Fs,order))
    return data


def __opto_get_all_lfp_analytic_quick_parallel_helper__():
    assert 0
    pass

def opto_get_all_lfp_analytic_quick_parallel(opto_dataset,fa,fb):
    '''
    Fs = 1000.0
    order = 4
    data = metaloadmat(opto_dataset+'_compact')['lfp']
    data = data.transpose((0,2,1))
    data = squeeze(parmap(
        __opto_get_all_lfp_analytic_quick_parallel_helper__,
        None)
    data = bandfilter(hilbert(data),fa,fb,Fs,order)
    return data
    '''
    assert 0
    pass

def __opto_get_lfp_analytic_helper__(params):
    '''
    Parallel function wrapper for opto_get_lfp_analytic
    '''
    (i,data,fa,fb,Fs,order) = params
    print(5,i)
    return i,hilbert(bandfilter(data,fa,fb,Fs,order))

def opto_get_lfp_analytic(opto_dataset,channel,fa,fb,order=4):
    '''
    Retrieves channel or channels from opto LFP dataset
    Channels are 1-indexed

    Parameters
    ----------
    opto_dataset : string
        path or string identifier for a dataset
    channel:
        1-indexed channel ID or None to return a NTimes x NChannel array
        of all LFP data
    fa:
        low frequency of band-pass, or 'None' to use a low-pass filter.
        if fb is 'None' then this is the cutoff for a high-pass filter.
    fb:
        high-frequency of band-pass, or 'None to use a high-pass filter.
        if fa is 'None' then this is the cutoff for a low-pass filter
    '''
    Fs = opto_get_Fs(opto_dataset)
    if channel is None:
        data = metaloadmat(opto_dataset)['LFP'].T
        #return array([bandfilter(x,fa,fb,Fs,order) for x in data])
        problems = [(i,x,fa,fb,Fs,order) for i,x in enumerate(data)]
        data = array(parmap(__opto_get_lfp_analytic_helper__,problems,verbose=1))
        return squeeze(data)
    else:
        assert channel>=1
        data = metaloadmat(opto_dataset)['LFP'][:,channel-1]
        return hilbert(bandfilter(data,fa,fb,Fs,order))

@memoize
def opto_get_Fs(opto_dataset):
    return metaloadmat(opto_dataset)['Fs'][0,0]
