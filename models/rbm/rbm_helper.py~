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

from neurotools.nlab import *
from neurotools.models.rbm.rbm import *
from neurotools.models.rbm.rbm_sample import *

import neurotools.models.rbm.rbm as rb
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats  import entropy
from numpy.random import *

import glob
import itertools

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

from IPython.core.pylabtools import figsize
figsize(14, 7)

try:
    from scipy.misc import logsumexp
except:
    from scipy.special import logsumexp

def make_filename(RADIUS, BATCH, REG, COV=False, NUM='', prefix='../data'):
    # construct a file name template
    regstr = '_reg_'
    covstr = '_cov_' if COV else '_'
    DIR = '%s/cifar_small_%smulti%sbatch%s%s%s_%s'%(prefix,NUM,covstr,BATCH,regstr,REG,RADIUS)
    print('reading from',DIR)
    FILENAME = DIR+'/rbm_{}_nh{}_T{}.npz'
    return DIR, FILENAME

def get_trials(DIR):
    NHIDDENS = np.array([fn[fn.find('_nh') + 3:fn.rfind('_T')]
                         for fn in glob.glob(DIR + '/*para*')]).astype(int)
    NHIDDENS.sort()
    print('hidden units used in simulation: ', NHIDDENS)
    TEMPERATURES = ([float(fn[fn.find('_T') + 2:fn.rfind('.npz')])
                     for fn in glob.glob(DIR + '/*fim*_nh' + str(NHIDDENS[0]) + '_*')])
    TEMPERATURES.sort()
    print('temperatures used in simulation: ', TEMPERATURES)
    return NHIDDENS, TEMPERATURES

def scattercompare(x,y,xl='',yl='',
                   tit=None,
                   nbins=10,
                   idline=None,
                   adaptlimits=True,
                   meanline=True,
                   shadevariance=False):
    '''
    Plots scatter plot
    Estimates mean of dependent variable in histogram bins of dependent 
    variable
    Plots identity line as well
    '''
    y = np.array(y)
    x = np.array(x)
    plt.scatter(x,y,0.1,color=AZURE)
    if meanline:
        order = np.argsort(x)
        m  = neurotools.signal.box_filter(y[order],int(np.sqrt(len(x))))
        plt.plot(x[order],m,color=BLACK,lw=2.5)
        if shadevariance:
            mm = neurotools.signal.box_filter((y**2)[order],int(np.sqrt(len(x))))
            v  = mm - m*m
            s  = np.sqrt(v)
            e  = 1.96*s
            plt.fill_between(x[order],m-e,m+e,color=(0.1,)*4)
    neurotools.graphics.plot.simpleaxis()
    plt.xlabel(xl)
    plt.ylabel(yl)
    if tit is None: 
        tit = '%s vs. %s'%(yl,xl) if xl!='' and yl!='' else ''
    plt.title(tit)
    neurotools.graphics.plot.simpleaxis()
    xlim(np.min(x),np.max(x))
    if np.all(y>=0) and ylim()[0]<=0:
        ylim(0,ylim()[1])
    # Clever limits
    xlim(np.min(x),np.max(x))
    yl = ylim()
    if adaptlimits:
        xmax = round_to_precision(percentile(x,99),3)
        xmax = min(xmax,xlim()[1])
        xlim(xlim()[0],xmax)
        usey = y[x<=xmax]
        ymax = round_to_precision(percentile(usey,99),1)
        ylim(ylim()[0],ymax)
        yl = ylim()
    # Identity lines
    if idline is True or idline==1:
        t = linspace(*(xlim()+(10,)))
        plot(t,t,color=RUST,lw=2)
        ylim(*yl)
    nicey()
    fudgey(12)

def barcompare(x,y,xl='',yl='',
                   tit=None,
                   nbins=10,
                    skip=100,
                   idline=None,
                   adaptlimits=True,
                   meanline=True,
                   shadevariance=False):
    '''
    Plots bar plot
    Estimates mean of dependent variable in histogram bins of dependent 
    variable
    Plots identity line as well
    '''
    y = np.array(y)
    x = np.array(x)
    bins = array(sorted(x))[::skip]
    nbins = len(bins)-1
    means,stds,sems = [],[],[]
    Deltae = (bins[1:]+bins[:-1])*0.5
    for i in range(nbins):
        a = bins[i]
        b = bins[i+1]
        ok = (x>=a) & (x<b)
        get = y[ok]
        n = sum(ok)
        v = var(get)
        sem = sqrt(v/n)
        means.append(mean(get))
        stds.append(sqrt(v))
        sems.append(sem*1.96)
    mu  = array(means)
    sigma  = array(stds)
    dmu = array(sems)
    scatter(Deltae,mu,0.1)
    plt.errorbar(Deltae, mu, sigma, fmt='.', markersize=4, lw=1, label=u'Observations',zorder=inf)
    plot(bins,bins,color=RUST)
    ylim(0,20)
    xlim(5,20)
    nicey()
    simpleaxis()
    
    if meanline:
        order = np.argsort(x)
        m  = neurotools.signal.box_filter(y[order],int(np.sqrt(len(x))))
        plt.plot(x[order],m,color=BLACK,lw=0.85)
        if shadevariance:
            mm = neurotools.signal.box_filter((y**2)[order],int(np.sqrt(len(x))))
            v  = mm - m*m
            s  = np.sqrt(v)
            e  = 1.96*s
            plt.fill_between(x[order],m-e,m+e,color=(0.1,)*4)
        
    neurotools.graphics.plot.simpleaxis()
    plt.xlabel(xl)
    plt.ylabel(yl)
    if tit is None: 
        tit = '%s vs. %s'%(yl,xl) if xl!='' and yl!='' else ''
    plt.title(tit)
    neurotools.graphics.plot.simpleaxis()
    xlim(np.min(x),np.max(x))
    if np.all(y>=0) and ylim()[0]<=0:
        ylim(0,ylim()[1])
    # Clever limits
    xlim(np.min(x),np.max(x))
    yl = ylim()
    if adaptlimits:
        xmax = round_to_precision(percentile(x,99),3)
        xmax = min(xmax,xlim()[1])
        xlim(xlim()[0],xmax)
        usey = y[x<=xmax]
        ymax = round_to_precision(percentile(usey,99),1)
        ylim(ylim()[0],ymax)
        yl = ylim()
    # Identity lines
    if idline is True or idline==1:
        t = linspace(*(xlim()+(10,)))
        plot(t,t,color=RUST,lw=2)
        ylim(*yl)
    nicey()
    fudgey(12)
    
def zipfplot(Eh):
    '''
    Zipf's law plot for an energy distribution
    '''
    # PDF approach
    Eh  = np.array(sorted(Eh))
    HEh = -slog(np.diff(Eh))
    CDF = np.cumsum(Eh)
    HEh_smoothed = neurotools.signal.box_filter(HEh,int(sqrt(len(HEh))))
    plot(Eh[1:],HEh_smoothed,color=OCHRE,lw=2,label='PDF method')
    plot(Eh,log(CDF),color=AZURE,lw=2,label='CDF method')
    xlim(min(Eh),max(Eh))
    ylim(min(Eh),max(Eh))
    simpleaxis()
    t = linspace(*(xlim()+(10,)))
    plot(t,t,color=RUST,lw=1.5)
    xlabel('Energy')
    ylabel('Entropy')
    nice_legend()
    
    
    
