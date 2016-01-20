

opto_dataset = 'TOMMY_MI_121101_full_trial_continuous_square_pulse_6mW001'

import sys, os
sys.path.insert(1,os.path.expanduser('~/Dropbox/bin'))
import neurotools
from neurotools.nlab import *
from neurotools.tools import metaloadmat,globalize
nowarn()
#opto_data = metaloadmat(opto_dataset)
#print opto_data['README'][0]

@globalize
def opto_get_events_passive(opto_dataset):
    start,stop = metaloadmat(opto_dataset)['events']
    return start, stop

def opto_get_all_lfp_quick(opto_dataset):
    data = metaloadmat(opto_dataset+'_compact')['lfp']
    return data

@globalize
def opto_get_map(opto_dataset):
    return array(metaloadmat(opto_dataset+'_compact')['arrayChannelMap'])

@globalize
def opto_get_laser(opto_dataset):
    return metaloadmat(opto_dataset)['laser'][0]

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

def __opto_get_lfp_filtered_helper__((i,data,fa,fb,Fs,order)):
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

def __opto_get_lfp_analytic_helper__((i,data,fa,fb,Fs,order)):
    '''
    Parallel function wrapper for opto_get_lfp_analytic
    '''
    print 5,i
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
    print 1
    Fs = opto_get_Fs(opto_dataset)
    if channel is None:
        print 2
        data = metaloadmat(opto_dataset)['LFP'].T
        #return array([bandfilter(x,fa,fb,Fs,order) for x in data])
        print 3
        problems = [(i,x,fa,fb,Fs,order) for i,x in enumerate(data)]
        print 4
        data = array(parmap(__opto_get_lfp_analytic_helper__,problems,verbose=1))
        return squeeze(data)
    else:
        assert channel>=1
        data = metaloadmat(opto_dataset)['LFP'][:,channel-1]
        return hilbert(bandfilter(data,fa,fb,Fs,order))

@globalize
def opto_get_Fs(opto_dataset):
    return metaloadmat(opto_dataset)['Fs'][0,0]








