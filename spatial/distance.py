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

'''
Routines for computing statistics from multielectrode arrays
that are sensitive to distances between channels.

Test code
---------

    session='SPK120925'
    area='M1'
    am = get_array_map(session,area)
    print(am)

    for ch1 in get_good_channels(session,area):
        for ch2 in get_good_channels(session,area):
            print(ch1,ch2,get_pair_distance(session,area,ch1,ch2))

    trial = get_good_trials(session,area)[0]
    epoch = 6,-1000,6000

    cohere_pairs()
'''
'''
*X* is a *numSamples* * *numCols* array
*ij* is a list of tuples.  Each tuple is a pair of indexes into
the columns of X for which you want to compute coherence.  For
example, if *X* has 64 columns, and you want to compute all
nonredundant pairs, define *ij* as::
'''

# TODO: fix imports
#from matplotlib.mlab import *
#from collections import *

def get_electrode_locations(session,area):
    assert 0 # not implemented
    warn('returns a dictionary indexed by channel ID (1-indexed)')
    am = get_array_map(session,area)

def get_pair_distance(session,area,ch1,ch2):
    warn('using array map, not anatomical overlay')
    warn('channels should be 1-indexed')
    #locations = get_electrode_locations(session,area)
    #l1 = locations[ch1]
    #l2 = locations[ch2]
    #return abs(l1-l2)
    am = get_array_map(session,area,removebad=False)
    x1,y1 = where(am==ch1)
    x2,y2 = where(am==ch2)
    if not (len(x1)==1 and len(x2)==1):
        print('Something wrong, multiple or missing matches for electrodes')
        print(ch1,ch2,(x1,y1),(x2,y2))
        print(am)
        assert 0
    return sqrt( (x1-x2)**2 + (y1-y2)**2 )[0]

def distance_angular_deviation(session,area,trial,epoch,threads=1):
    '''
    Computes angular distance function over time, frequency, and distance.
    cos(theta-phi)
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if epoch is None: epoch = 6,-1000,6000
    lfp = get_all_raw_lfp(session,area,trial,epoch)
    freqs,cwt = fft_cwt_transposed(lfp,1,55,w=4.0,resolution=1,threads=threads)
    NCH,F,T = shape(cwt)
    h = angle(cwt)
    results  = defaultdict(list)
    channels = get_good_channels(session,area)
    for i,ch1 in enumerate(channels):
        for j,ch2 in enumerate(channels):
            if i>=j: continue
            hd = cos(h[i,:,:]-h[j,:,:])
            x = get_pair_distance(session,area,ch1,ch2)
            results[x].append(((ch1,ch2),hd))
    return results

def get_averaged_angular_distance(args):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    (session,area,trial,epoch) = args #py 2.x->3 compatibility
    if epoch is None: epoch = 6,-1000,6000
    lfp = get_all_raw_lfp(session,area,trial,epoch)
    freqs,cwt = fft_cwt_transposed(lfp,1,55,w=4.0,resolution=1,threads=1)
    NCH,F,T = shape(cwt)
    h = angle(cwt)
    channels = get_good_channels(session,area)
    sums   = {}
    counts = {}
    for i,ch1 in enumerate(channels):
        for j,ch2 in enumerate(channels):
            if i>=j: continue
            x = abs(cos(h[i,:,:]-h[j,:,:]))
            k = get_pair_distance(session,area,ch1,ch2)
            l = round(k)
            if l in sums:
                sums[l]   += x
                counts[l] += 1
            else:
                sums[l] = arr(x)
                counts[l] = 1
    for k in sums.keys():
        x = sums[k]
        x = x/float(counts[k])
        x = mean_block(x,50)
        sums[k] = x
    blocked = zeros((len(sums.keys()),)+shape(sums[k]),dtype=float64)
    for k,v in sums.iteritems():
        blocked[k-1,:] = v
    del sums
    return trial, blocked

def get_length_constant_trial(session,area,trial,epoch):
    if epoch is None: epoch = 6,-1000,6000
    _, x = get_averaged_angular_distance((session,area,trial,epoch))
    z = -log(x[:7,:,:]+1e-6)
    q = ((z.T/arange(1,1+7)).T)**(-1)
    e = mean(q,0)
    return e
    #return mean(diff(x[:7,:,:],1,0)/0.4,0)

def get_average_synchrony_length_constant_parallel(session,area,epoch):
    '''
    TODO: move this to the CGID project or modify to be more general  
    '''
    '''
    Test code
    ---------
    session='SPK120925'
    session='RUS120518'
    area='PMv'
    freqs,y = get_average_synchrony_length_constant_parallel(session,area,epoch)
    imshow(y,extent=(0,7000,freqs[-1],freqs[0]),
        interpolation='nearest',aspect=50,cmap=luminance)#,vmin=0.5,vmax=0.7)

    Dev code
    from os.path import *
    execfile(expanduser('~/Dropbox/bin/cgid/cgidsetup.py'))
    session='SPK120925'
    area='M1'
    am = get_array_map(session,area)
    print(am)
    for ch1 in get_good_channels(session,area):
        for ch2 in get_good_channels(session,area):
            print(ch1,ch2,get_pair_distance(session,area,ch1,ch2))
    trial = get_good_trials(session,area)[0]
    epoch = 6,-1000,6000
    nowarn()
    def get_averaged_angular_distance(trial):
        print("processing trial",trial)
        results = distance_angular_deviation(session,area,trial,epoch,threads=1)
        sums   = {}
        counts = {}
        for k,v in results.iteritems():
            l = round(k)
            print(k,l)
            for ch,x in v:
                if l in sums:
                    sums[l]   += x
                    counts[l] += 1
                else:
                    sums[l] = arr(x)
                    counts[l] = 1
        for k in sums.keys():
            x = sums[k]
            x = x/float(counts[k])
            x = mean_block(x,50)
            sums[k] = x
        blocked = zeros((len(sums.keys()),)+shape(sums[k]),dtype=float64)
        for k,v in sums.iteritems():
            blocked[k-1,:] = v
        del results
        del sums
        return trial, blocked
    reset_pool()
    allblocked = squeeze(arr(parmap(get_averaged_angular_distance,get_good_trials(session,area))))
    freqs = arange(1,56)
    x = mean(allblocked,0)
    y = mean(diff(x[:7,:,:],0),0)
    use = (freqs>10)*(freqs<45)==1
    imshow(y[use,:],extent=(0,7000,freqs[use][-1],freqs[use][0]),
        interpolation='nearest',aspect=50,cmap=luminance,vmin=0.5,vmax=0.7)
    '''
    warn('Frequency range is hard coded 1..55 in the wavelet transform')
    warn('But we focus on 10-45Hz')
    warn('only looks over 2.4mm')
    if epoch is None: epoch = 6,-1000,6000
    # am is not used, but should preload dataset
    # to avoid memory distasters when parallizing
    am = get_array_map(session,area)
    jobs = [(session,area,trial,epoch) for trial in get_good_trials(session,area)]
    reset_pool()
    allblocked = squeeze(arr(parmap(get_averaged_angular_distance,jobs)))
    freqs = arange(1,56)
    x = mean(allblocked,0)
    #y = mean(diff(x[:7,:,:],1,0),0)
    #return freqs,y
    #use = (freqs>10)*(freqs<45)==1
    #return freqs[use],y[use,:]
    z = -log(x[:7,:,:])
    q = ((z.T/arange(1,1+7)).T)**(-1)
    e = mean(q,0)
    return freqs,e

def synchrony_length_constant_areas_summary(session,trial,epoch):
    '''
    TODO: move this to the CGID project or modify to be more general
    '''
    '''
    session = 'SPK120924'
    synchrony_length_constant_areas_summary(session,good_trials(session)[0],None)
    '''
    res = {}
    for area in areas:
        print(session,area)
        res[area]=get_length_constant_trial(session,area,trial,epoch)

    freqs = arange(1,56)
    figure('Length constant',figsize=(5.5,3.7))
    clf()

    vmin,vmax=-0.5,0.5
    vmin,vmax=0,100

    def dz(x):
        x[isnan(x)]=0
        return x
    dz = piper(dz)
    toshow = (freqs>=0)*(freqs<=45)*ok==1

    ax1 = subplot2grid((3,1),(0,0))
    CWT = dz|res['M1']
    im = imshow(CWT[toshow,:],aspect=aspect,
        extent=(0,shape(CWT)[1],freqs[toshow][0],freqs[toshow][-1]),
        interpolation='nearest',
        origin='lower',
        vmin=vmin,vmax=vmax,
        cmap=COLORMAP)
    ylabel('Hz')
    xticks([])
    ylim(freqs[toshow][0],freqs[toshow][-1])
    yticks([2,15,30,45])
    title('Area M1')

    ax2 = subplot2grid((3,1),(1,0))
    CWT = dz|res['PMv']
    im = imshow(CWT[toshow,:],aspect=aspect,
        extent=(0,shape(CWT)[1],freqs[toshow][0],freqs[toshow][-1]),
        interpolation='nearest',
        origin='lower',
        vmin=vmin,vmax=vmax,
        cmap=COLORMAP)
    ylabel('Hz')
    xticks([])
    set_cmap(COLORMAP)
    ylim(freqs[toshow][0],freqs[toshow][-1])
    yticks([2,15,30,45])
    title('Area PMv')

    ax3 = subplot2grid((3,1),(2,0))
    CWT = dz|res['PMd']
    im = imshow(CWT[toshow,:],aspect=aspect,
        extent=(0,shape(CWT)[1],freqs[toshow][0],freqs[toshow][-1]),
        interpolation='nearest',
        origin='lower',
        vmin=vmin,vmax=vmax,
        cmap=COLORMAP)
    ylabel('Hz')
    xlabel('Time (ms)')
    set_cmap(COLORMAP)
    ylim(freqs[toshow][0],freqs[toshow][-1])
    yticks([2,15,30,45])
    title('Area PMd')

    suptitle('LFP cross-trial phase synchrony, %s'%session,fontsize=15)
    tight_layout()
    subplots_adjust(top=.86)
    sca(ax1)
    overlayEvents(FS=1,fontsize=10,npad=1)
    noaxis()
    sca(ax2)
    overlayEvents(FS=1,fontsize=10,npad=1)
    noaxis()
    sca(ax3)
    overlayEvents('w','k',FS=1,fontsize=10,npad=1)
    noaxis()
