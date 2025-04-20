#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import scipy.stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorsys

from mpl_toolkits import axes_grid1
from neurotools.graphics.color import hcl2rgb,hsv2rgb

# Configure matplotlib defaults
def configure_matplotlib():
    global TEXTWIDTH,SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE
    matplotlib.rcParams['figure.dpi']=300
    TEXTWIDTH = 5.62708
    matplotlib.rcParams['figure.figsize'] = (TEXTWIDTH, TEXTWIDTH/sqrt(2))
    SMALL_SIZE  = 7
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 9
    matplotlib.rc('font'  , size     =SMALL_SIZE ) # controls default text sizes
    matplotlib.rc('axes'  , titlesize=MEDIUM_SIZE) # fontsize of the axes title
    matplotlib.rc('axes'  , labelsize=MEDIUM_SIZE) # fontsize of the x and y labels
    matplotlib.rc('xtick' , labelsize=SMALL_SIZE ) # fontsize of the tick labels
    matplotlib.rc('ytick' , labelsize=SMALL_SIZE ) # fontsize of the tick labels
    matplotlib.rc('legend', fontsize =SMALL_SIZE ) # legend fontsize
    matplotlib.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title
    matplotlib.rc('lines' , solid_capstyle='round')

def add_circle(radius=1,lw=0.75,color='k',linestyle='-',n=360):
    '''
    
    Other Parameters
    ----------------
    radius:1
    lw:0.75
    color:'k'
    linestyle:'-'
    n:360
    '''
    circle = np.exp(1j*np.linspace(0,2*np.pi,n))*radius
    plt.plot(circle.real,circle.imag,lw=lw,color=color,linestyle=linestyle)

def label_complex_axes(color='k'):
    '''
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    plt.text(0,plt.ylim()[1],r'$\Im$',va='top',color=color)
    plt.text(plt.xlim()[1],0,r'$\Re$',va='bottom',ha='right',color=color)
    
def add_complex_axes(lw=0.75,color='w',limits=[(-2,2),(-2,2)]):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    plt.plot(limits[0],[0,0],lw=lw,color=color,linestyle=':')
    plt.plot([0,0],limits[1],lw=lw,color=color,linestyle=':')
    add_circle(1,lw,color,linestyle=':')
    plt.xlim(*limits[0])
    plt.ylim(*limits[1])
    label_complex_axes(color)
    plt.gca().spines['top'  ].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
# Colorbar Patch from here
# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def phase_magnitude_figure(z,limit=2):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    plt.subplot(121)
    # The $\tan^{-1}|z|$ scaling on the magnitude is arbitrary,
    # and done only to map ${0,\infty)$ to a finite color range
    ext=(-limit,limit,-limit,limit)
    v = np.arctan(abs(z))/(np.pi/2)
    v = v*0.975+0.0125
    v = v**0.5
    im = plt.imshow(v,extent=ext,cmap='bone')
    plt.xlabel(r'$\Re$')
    plt.ylabel(r'$\Im$')
    plt.title('Magnitude')
    add_complex_axes(limits=((-limit,limit),)*2)
    plt.subplot(122)
    im = plt.imshow(np.angle(z),extent=ext,cmap='hsv',interpolation='nearest')
    plt.xlabel(r'$\Re$')
    plt.ylabel(r'$\Im$')
    cax = plt.add_colorbar(im,label="Phase $\\theta$")
    cax.set_ticks([-np.pi*0.999,0,np.pi*0.999])
    cax.ax.set_yticklabels(['-π','0','π'])
    plt.title('Phase')
    add_complex_axes(limits=((-limit,limit),)*2)
    plt.tight_layout()

def joint_phase_magnitude_plot(z,
    limit=[(-2,2),(-2,2)],
    color='w',
    draw_complex_axes=True,
    autoscale=False,
    nonlinearscale=True,
    **opts):
    '''
    
    Parameters
    ----------
    z: 2D complex-valued array
    
    Other Parameters
    ----------------
    limit: 
        Plot limits as [(real_min,real_max),(imag_min,imag_max)]
    color:
        Matplotlib color specifier for annotations
    draw_complex_axes:
        Overlay axes atop plot?
    autoscale:
        Automatically re-center magniture scale. 
    nonlinearscale:
        Use arctan magnitude mapping to compress possibly infinite 
        dynamic range
    '''
    if nonlinearscale:
        if autoscale: 
            z /= sqrt(np.mean(abs(z)**2))
        v = np.arctan(abs(z))/(np.pi/2)
    else:
        z /= np.max(abs(z))
        v = np.abs(z)
    
    h = ((np.angle(z)+np.pi)%(2*np.pi))/(2*np.pi)
    v[np.isnan(v)]=1
    h[np.isnan(h)]=0
    
    # Original flavor
    v = v*0.975+0.0125
    rgb = [colorsys.hsv_to_rgb(hi,min(1,2*(1-vi))**0.5,min(1,2*vi)**0.5) \
        for (hi,vi) in zip(h.ravel(),v.ravel())]
    
    #rgb = [colorsys.hsv_to_rgb(hi,min(1,(1-vi)),min(1,vi)) \
    #    for (hi,vi) in zip(h.ravel(),v.ravel())]
    
    #v = v*0.9+0.05
    #rgb = [hsv2rgb(hi*360,min(1,2*(1-vi))**0.5,min(1,2*vi)**0.5,force_luminance=vi) \
    #    for (hi,vi) in zip(h.ravel(),v.ravel())]
    
    #v = v*0.9+0.05
    #rgb = [hcl2rgb(hi*360,min(1,2*(1-vi))**0.5,min(1,2*vi)**0.5,target=vi) \
    #    for (hi,vi) in zip(h.ravel(),v.ravel())]
    
    opts = dict(interpolation='bicubic')|opts
        
    rgb = np.array(rgb).reshape((z.shape[0],z.shape[1],3))
    plt.imshow(rgb,extent=tuple(limit[0])+tuple(limit[1]),**opts)
    if draw_complex_axes:
        add_complex_axes(limits=limit,color=color)
    plt.title('Phase-magnitude plot')
    plt.xticks([limit[0][0],0,limit[0][1]])
    plt.yticks([limit[1][0],0,limit[1][1]])


def do_bode_z(ω,Y,ax1,ax2,stitle='',label=None,color='k',linestyle='-',lw=0.8,drawlegend=True):
    '''
    Parameters
    ----------
    ω: frequencies at which to evaluate
    w: complex-valued output of z-transfer function at each frequency
    ax1: axis for the magnitude (gain) plot
    ax2: axis for phase plot
    '''
    w = Y(np.exp(1j*ω))
    if label is None:
        label = stitle
    plt.sca(ax1)
    semilogy(ω,(np.abs(w)),label=label,color=color,linestyle=linestyle,lw=lw)
    ylabel('Gain')
    def do_axis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.autoscale(enable=True, axis='x', tight=True)
        for text in ax.get_xminorticklabels():
            text.set_rotation(90)
    do_axis(plt.gca())
    plt.xticks([0,np.pi],['',''])
    plt.title(stitle)
    if drawlegend:
        lg = legend(loc='center left',bbox_to_anchor=(1,0.5))
        lg.get_frame().set_linewidth(0.0)
    plt.sca(ax2)
    θ = np.angle(w)#%(2*np.pi)-2*np.pi
    plt.plot(ω,θ,label=label,color=color,linestyle=linestyle,lw=lw)
    plt.ylabel('Phase')
    do_axis(gca())
    plt.yticks([-np.pi,0,np.pi],['-π','0','π'])
    plt.ylim(-np.pi,np.pi)
    plt.xticks([0,np.pi/2,np.pi],['0','π/2','π'])
    plt.axhline(0,lw=0.5,color='k',linestyle=':')
    plt.xlabel('Normalized frequency')

def do_nyquist(ax,z,Y,extent=8):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    ζ = Y(z)
    x,y = ζ.real,ζ.imag
    plt.sca(ax)
    plt.plot(x,y)
    add_complex_axes(color='k',limits = ((-extent,extent),(-extent,extent)))
    plt.title('Nyquist plot')
    plt.xlabel(r'$\Re$')
    plt.ylabel(r'$\Im$')
    
def do_bode_and_nyquist(Y,extent=4):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    plt.figure(figsize=(TEXTWIDTH,2.75))
    ax1 = plt.subplot2grid((2,4),(0,0),rowspan=1,colspan=2)
    ax2 = plt.subplot2grid((2,4),(1,0),rowspan=1,colspan=2)
    ax3 = plt.subplot2grid((2,4),(0,2),rowspan=2,colspan=2)
    ω = np.linspace(0,np.pi,1000)
    do_bode_z(ω,Y,ax1,ax2,stitle='Bode plot',lw=1.5,drawlegend=False)
    z = np.exp(1j*linspace(0,2*np.pi,10000))
    do_nyquist(ax3,z,Y,extent=extent)
    plt.subplots_adjust(left=0.125,bottom=0.15,top=0.9,right=0.975,wspace=1,hspace=0.55)
    return ax1,ax2,ax3
    
def do_bode_s(ω,w,ax1,ax2,stitle='',label=None,color='k',linestyle='-',lw=0.8):
    '''
    Parameters
    ----------
    ω : frequencies at which to evaluate
    w : complex-valued output of z-transfer function at each frequency
    ax1 : axis for the magnitude (gain) plot
    ax2 : axis for phase plot
    '''
    if label is None:
        label = stitle
    plt.sca(ax1)
    plt.loglog(ω,(abs(w)),label=label,color=color,linestyle=linestyle,lw=lw)
    plt.ylabel('Gain')
    def do_axis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.autoscale(enable=True, axis='x', tight=True)
    do_axis(plt.gca())
    plt.title(stitle)
    lg = plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    lg.get_frame().set_linewidth(0.0)
    plt.sca(ax2)
    plt.semilogx(ω,np.angle(w)%(2*np.pi)-2*np.pi,label=label,color=color,linestyle=linestyle,lw=lw)
    plt.ylabel('Phase')
    do_axis(plt.gca())
    plt.yticks([-2*np.pi,-np.pi,0],['-2π','-π','0'])
    plt.ylim(-2*np.pi,0)


