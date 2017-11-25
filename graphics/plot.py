#!/usr/bin/python
# -*- coding: UTF-8 -*-
# BEGIN PYTHON 2/3 COMPATIBILITY BOILERPLATE
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
import sys
if sys.version_info<(3,):
    from itertools import imap as map
# END PYTHON 2/3 COMPATIBILITY BOILERPLATE

from   neurotools.graphics.color   import *
from   neurotools.getfftw import *
from   neurotools.tools   import *
import os
import pickle
import scipy
import numpy
import scipy.optimize
from   scipy.io          import savemat
from   scipy.optimize    import leastsq
from   multiprocessing   import Process, Pipe, cpu_count, Pool
from   scipy.io          import loadmat
from   scipy.signal      import butter,filtfilt,lfilter
from   matplotlib.pyplot import *
import matplotlib.pyplot as plt

try: # python 2.x
    from itertools import izip, chain
except: # python 3
    from itertools import chain
    izip = zip

try:
    import statsmodels
    import statsmodels.api as smapi
    import statsmodels.graphics as smgraphics
except:
    print('could not find statsmodels; some plotting functions missing')

def zscore(x):
    '''
    Parameters
    ----------
    x : data to zscore
    
    Returns
    -------
    x : z-scored copy of x
    '''
    return (x-np.mean(x,0))/np.std(x,0)

def simpleaxis(ax=None):
    '''
    Only draw the bottom and left axis lines
    
    Parameters
    ----------
    ax : optiona, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def simpleraxis(ax=None):
    '''
    Only draw the left y axis, nothing else
    
    Parameters
    ----------
    ax : optiona, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def noaxis(ax=None):
    '''
    Hide all axes
    
    Parameters
    ----------
    ax : optiona, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def nicebp(bp,c):
    '''
    Improve the appearance of a box and whiskers plot
    
    Parameters
    ----------
    bp : point to boxplot object returned by matplotlib
    c : color to set boxes to
    '''
    for kk in 'boxes whiskers fliers caps'.split():
        setp(bp[kk], color=c)
    setp(bp['whiskers'], linestyle='solid',linewidth=.5)
    setp(bp['caps'],     linestyle='solid',linewidth=.5)
    #setp(bp['caps'], color=(0,)*4)

# printing routines


def percent(n,total):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    return '%0.2g%%'%(n*100.0/total)
    
def shortscientific(x,prec=0):
    '''
    Parameters
    ----------
    x : scalar numeric
    prec : non-negative integer
    
    Returns
    -------
    '''
    return ('%.*e'%(prec,x)).replace('-0','-').replace('+','').replace('e0','e')

def eformat(f, prec, exp_digits):
    '''
    Format a float in scientific notation with fewer characters.
    
    Parameters
    ----------
    f: scalar
        Number
    prec:
        Precision
    exp_digits: int
        Number exponent digits
    
    Returns
    -------
    string : reformatted string
    '''
    if not np.isfinite(f):
        return '%e'%f
    s = "%.*e"%(prec, f)
    mantissa, exponent = s.split('e')
    exponent = int(exponent)
    s = mantissa + "e%+0*d"%(exp_digits+1, exponent)
    s = s.replace('+','')
    return s

def nicey():
    '''
    Mark only the min/max value of y axis
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ylim()[0]<0:
        yticks([ylim()[0],0,ylim()[1]])
    else:
        yticks([ylim()[0],ylim()[1]])

def nicex():
    '''
    Mark only the min/max value of x axis
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if xlim()[0]<0:
        xticks([xlim()[0],0,xlim()[1]])
    else:
        xticks([xlim()[0],xlim()[1]])

def nicexy():
    '''
    Mark only the min/max value of y/y axis
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    nicey()
    nicex()

def positivex():
    '''
    Sets the lower x limit to zero, and the upper limit to the largest
    positive value un the current xlimit. If the curent xlim() is
    negative, a value error is raised.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    top = np.max(xlim())
    if top<=0:
        raise ValueError('Current axis view lies within negative '+
            'numbers, cannot crop to a positive range')
    xlim(0,top)
    nicex()

def positivey():
    '''
    Sets the lower y limit to zero, and the upper limit to the largest
    positive value un the current ylimit. If the curent ylim() is
    negative, a value error is raised.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    top = np.max(ylim())
    if top<=0:
        raise ValueError('Current axis view lies within negative '+
            'numbers, cannot crop to a positive range')
    ylim(0,top)
    nicey()

def positivexy():
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    positivex()
    positivey()

def xylim(a,b,ax=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax==None: ax = plt.gca()
    ax.set_xlim(a,b)
    ax.set_ylim(a,b)

def nox():
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    xticks([])
    xlabel('')

def noy():
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    yticks([])
    ylabel('')

def righty(ax=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax==None: ax=plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

def unity():
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    ylim(0,1)
    nicey()

def unitx():
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    xlim(0,1)
    nicex()

def force_aspect(aspect=1,a=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if a is None: a = plt.gca()
    x1,x2=a.get_xlim()
    y1,y2=a.get_ylim()
    a.set_aspect(np.abs((x2-x1)/(y2-y1))/aspect)

def unitaxes(a=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if a is None: a=plt.gca()
    a.set_xlim(0,1)
    a.set_ylim(0,1)
    a.set_xticks([0,1])
    a.set_yticks([0,1])
    a.set_xticklabels(['0','1'])
    a.set_yticklabels(['0','1'])

def adjustmap(arraymap):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    nrow,ncol = shape(arraymap)
    adjustedmap = np.array(arraymap)
    available   = sorted(list(set([x for x in ravel(arraymap) if x>0])))
    for i,ch in enumerate(available):
        adjustedmap[arraymap==ch]=i
    return adjustedmap

def get_ax_size(ax=None,fig=None):
    '''
    Gets tha axis size in figure-relative units
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax is None: ax  = plt.gca()
    '''http://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels'''
    fig  = plt.gcf()
    ax   = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width  *= fig.dpi
    height *= fig.dpi
    return width, height

def get_ax_pixel(ax=None,fig=None):
    '''
    Gets tha axis size in pixels
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax is None: ax  = plt.gca()
    # w/h in pixels
    w,h = get_ax_size()
    # one px in axis units is the axis span div no. pix
    dy = diff(ylim())
    dx = diff(xlim())
    return dx/float(w),dy/float(h)

def pixels_to_xunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current x-axis
    scale
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dx  = np.diff(plt.xlim())[0]
    return n*dx/float(w)

def yunits_to_pixels(n,ax=None,fig=None):
    '''
    Converts a measurement in units of the current y-axis to pixels
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dy  = np.diff(plt.ylim())[0]
    return n*float(h)/dy

def xunits_to_pixels(n,ax=None,fig=None):
    '''
    Converts a measurement in units of the current x-axis to pixels
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dx = diff(xlim())[0]
    return n*float(w)/dx

def pixels_to_yunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current y-axis
    scale
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dy = np.diff(ylim())[0]
    return n*dy/float(h)

def pixels_to_xfigureunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current
    figure width scale
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w_pixels = fig.get_size_inches()[0]*fig.dpi
    return n/float(w_pixels)

def pixels_to_yfigureunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current
    figure height scale
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    h_pixels = fig.get_size_inches()[1]*fig.dpi
    return n/float(h_pixels)

def adjust_ylabel_space(n,ax=None):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax=plt.gca()
    ax.yaxis.labelpad = n

def adjust_xlabel_space(n,ax=None):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax=plt.gca()
    ax.xaxis.labelpad = n

def nudge_axis_y_pixels(dy,ax=None):
    '''
    moves axis dx pixels.
    Direction of dx may depent on axis orientation. TODO: fix this
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = -pixels_to_yfigureunits(float(dy),ax)
    ax.set_position((x,y-dy,w,h))

def adjust_axis_height_pixels(dy,ax=None):
    '''
    moves axis dx pixels.
    Direction of dx may depent on axis orientation. TODO: fix this
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    ax.set_position((x,y,w,h-pixels_to_yfigureunits(float(dy),ax)))

def nudge_axis_y(dy,ax=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,ax)
    ax.set_position((x,y+dy,w,h))

def nudge_axis_x(dx,ax=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,ax)
    ax.set_position((x+dx,y,w,h))

def expand_axis_y(dy,ax=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,ax)
    ax.set_position((x,y,w,h+dy))

def nudge_axis_baseline(dy,ax=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,ax)
    ax.set_position((x,y+dy,w,h-dy))

def nudge_axis_left(dx,ax=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,ax)
    ax.set_position((x+dx,y,w-dx,h))

def zoombox(ax1,ax2,xspan1=None,xspan2=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    # need to do this to get the plot to ... update correctly
    show()
    draw()
    show()
    fig = plt.gcf()

    if xspan1==None:
        xspan1 = ax1.get_xlim()
    if xspan2==None:
        xspan2 = ax2.get_xlim()

    transFigure = fig.transFigure.inverted()
    coord1 = transFigure.transform(ax1.transData.transform([xspan1[0],ax1.get_ylim()[1]]))
    coord2 = transFigure.transform(ax2.transData.transform([xspan2[0],ax2.get_ylim()[0]]))
    line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                   transform=fig.transFigure,lw=1,color='k')
    fig.lines.append(line)
    coord1 = transFigure.transform(ax1.transData.transform([xspan1[1],ax1.get_ylim()[1]]))
    coord2 = transFigure.transform(ax2.transData.transform([xspan2[1],ax2.get_ylim()[0]]))
    line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                   transform=fig.transFigure,lw=1,color='k')
    fig.lines.append(line)
    show()

def fudgex(by=10,ax=None,doshow=False):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax=plt.gca()
    ax.xaxis.labelpad = -by
    plt.draw()
    if doshow:
        plt.show()

def fudgey(by=20,ax=None,doshow=False):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax=plt.gca()
    ax.yaxis.labelpad = -by
    plt.draw()
    if doshow:
        plt.show()

def fudgexy(by=10,ax=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    fudgex(by,ax)
    fudgey(by,ax)

def shade_edges(edges,color=(0.5,0.5,0.5,0.5)):
    '''
    Edges of the form (start,stop)
    Shades regions of graph defined by "edges"
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    a,b = ylim()
    c,d = xlim()
    for x1,x2 in zip(*edges):
        print(x1,x2)
        fill_between([x1,x2],[a,a],[b,b],color=color,lw=0)
    ylim(a,b)
    xlim(c,d)

shade = shade_edges

def ybar(x,**kwargs):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    a,b = ylim()
    plot([x,x],[a,b],**kwargs)
    ylim(a,b)

def xbar(y,**kwargs):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    a,b = xlim()
    plot([a,b],[y,y],**kwargs)
    xlim(a,b)

def allnice():
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    nicex()
    nicey()
    nice_legend()

def ybartext(x,t,c1,c2,**kwargs):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    a,b = ylim()
    outline = False
    if 'outline' in kwargs:
        outline = kwargs['outline']
        del kwargs['outline']
    plot([x,x],[a,b],**kwargs)
    ylim(a,b)
    dx,dy = get_ax_pixel()
    if outline:
        for ix in arange(-2,3)*dx:
            for iy in arange(-2,3)*dy:
                text(ix+x,iy+ylim()[1]-dy*4,t,
                    rotation=90,
                    color=c2,
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=12)
    text(x,ylim()[1]-dy*2,t,
        rotation=90,color=c1,fontsize=12,
        horizontalalignment='right',verticalalignment='top')
    ylim(a,b)

def xbartext(y,t,c1,c2,**kwargs):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    a,b = xlim()
    outline = False
    if 'outline' in kwargs:
        outline = kwargs['outline']
        del kwargs['outline']
    text_kwargs = {}
    text_kwargs['horizontalalignment'] = 'left'
    text_kwargs['verticalalignment'] = 'bottom'
    text_kwargs['fontsize'] = 12
    for key in text_kwargs.keys():
        if key in kwargs:
            text_kwargs[key] = kwargs[key]
            del kwargs[key]
    text_kwargs['color']=c1
    plot([a,b],[y,y],**kwargs)
    xlim(a,b)
    dx,dy = get_ax_pixel()
    if outline:
        for ix in arange(-2,3)*dx:
            for iy in arange(-2,3)*dy:
                text(ix+a+dx*4,iy+y,t,
                    color=c2,
                    horizontalalignment='left',verticalalignment='bottom',fontsize=12)

    if text_kwargs['horizontalalignment']=='left':
        # left aligned text
        text(a+dx*4,y,t,**text_kwargs)
    elif text_kwargs['horizontalalignment']=='right':
        # right aligned text
        text(b-dx*4,y,t,**text_kwargs)
    xlim(a,b)

def nice_legend(*args,**kwargs):
    '''
    Better defaults for the plot legend. TODO: make this into a
    matplotlib style.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    defaults = {
        'framealpha':0.9,
        'fancybox':True,
        'fontsize':10,
        'numpoints':1,
        'scatterpoints':1}
    defaults.update(kwargs)
    lg = legend(*args,**defaults)
    lg.get_frame().set_linewidth(0.0)
    return lg

def rangeto(rangefun,data):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    rangefun(np.min(data),np.max(data))

def rangeover(data):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    return np.min(data),np.max(data)

def cleartop(x):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    subplots_adjust(top=1-x)

def plotCWT(ff,cwt,aspect='auto',
    vmin=None,vmax=None,cm='afmhot',interpolation='nearest',dodraw=1):
    '''
    Parameters
    ----------
    ff : numeric
        frequencies
    cwt : numeric
        wavelet transformed data (what orientation?)
    '''
    cwt = squeeze(cwt)
    nf,N = shape(cwt)
    pwr    = np.abs(cwt)
    fest   = ff[argmax(pwr,0)]
    cla()
    imshow(pwr,aspect=aspect,extent=(0,N,ff[-1],ff[0]),vmin=vmin,vmax=vmax,interpolation=interpolation,cmap=cm)
    xlim(0,N)
    ylim(ff[0],ff[-1])
    try:
        tight_layout()
    except:
        print('tight_layout missing, how old is your python? seriously')
    if dodraw:
        draw()
        show()

def complex_axis(scale):
    '''
    Draws a nice complex-plane axis with LaTeX Re, Im labels and everything
    Parameters
    ----------
    
    Returns
    -------
    '''
    xlim(-scale,scale)
    ylim(-scale,scale)
    nicexy()
    ybartext(0,'$\Im(z)$','k','w',lw=1,color='k',outline=False)
    xbartext(0,'$\Re(z)$','k','w',lw=1,color='k',outline=False,horizontalalignment='right')
    noaxis()
    xlabel(u'μV',fontname='DejaVu Sans',fontsize=12)
    ylabel(u'μV',fontname='DejaVu Sans',fontsize=12)
    xticks(xticks()[0],fontsize=12)
    yticks(yticks()[0],fontsize=12)
    force_aspect()

def plotWTPhase(ff,cwt,aspect=None,ip='nearest'):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    cwt = squeeze(cwt)
    nf,N = shape(cwt)
    if aspect is None: aspect = N/float(nf)*0.5
    pwr = np.abs(cwt)
    rgb = complexHLArr2RGB(cwt*(0.9/nmx(pwr)))
    cla()
    imshow(rgb,cmap=None,aspect=aspect,
        extent=(0,N,ff[-1],ff[0]),interpolation=ip)
    xlim(0,N)
    ylim(ff[0],ff[-1])
    try:
        tight_layout()
    except:
        print('tight_layout missing, you should update matplotlib')
    draw()
    show()

wtpshow = plotWTPhase

def plotWTPhaseFig(ff,cwt,aspect=50,
    vmin=None,vmax=None,cm='bone',interpolation='nearest'):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    cwt = squeeze(cwt)
    nf,N = shape(cwt)
    pwr    = np.abs(cwt)
    fest   = ff[argmax(pwr,0)]
    clf()
    subplot(211)
    imshow(pwr,aspect=aspect,extent=(0,N,ff[-1],ff[0]),
        vmin=vmin,vmax=vmax,cmap=cm,interpolation=interpolation)
    xlim(0,N)
    ylim(ff[0],ff[-1])
    subplot(212)
    imshow(angle(cwt),aspect=aspect,extent=(0,N,ff[-1],ff[0]),
        vmin=vmin,vmax=vmax,cmap=medhue,interpolation=interpolation)
    xlim(0,N)
    ylim(ff[0],ff[-1])
    try:
        tight_layout()
    except:
        print('tight_layout missing, you should update')
    draw()
    show()

def domask(*args):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if len(args)>2:
        return (args[1],)+domask(args[0],*args[2:])
    mm = np.array(args[1])
    ok = ~args[0]
    N  = len(ok)
    M  = len(mm)
    if M<N:
        warn('WARNING MASK IS TOO LONG.')
        warn('MIGHT BE AN OFF BY 1 ERROR HERE')
        d = (N-M)/2
        print(len(mm),len(ok[d:N-((N-M)-d)]))
        mm[ok[d:N-((N-M)-d)]]=NaN
        return mm
    mm[ok]=NaN
    return mm

def fsize(f=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if f is None: f=plt.gcf()
    return f.get_size_inches()

# http://stackoverflow.com/questions/27826064/matplotlib-make-legend-keys-square
from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
class HandlerSquare(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width,height, fontsize, trans):
        '''
        Parameters
        ----------
        
        Returns
        -------
        '''
        center = xdescent + 0.5 * (width-height),ydescent
        p = mpatches.Rectangle(xy=center,width=height,height=height, angle=0.0)
        self.update_prop(p,orig_handle,legend)
        p.set_transform(trans)
        return [p]

def plot_complex(z,vm=None,aspect='auto',ip='bicubic',
    extent=None,onlyphase=False,previous=None,origin='lower'):
    '''
    Renders complex np.array as image, in polar form with magnitude mapped to
    lightness and hue mapped to phase.

    :param z: 2D np.array of complex values
    :param vm: max complex modulus. Default of None will use max(np.abs(z))
    :param aspect: image aspect ratio. defaults auto
    :param ip: interpolation mode to forward to imshow. defaults bicubic
    :param extent: extents (dimensions) for imshow. defaults None.
    :param previous: if available, output of a previous call to plot_complex
        can be used to skirt replotting nuisances and update data directly,
        which should be a little faster.

    :return img: a copy of the result of imshow, which can be reused in
        subsequent calls to speed things up, if animating a video
    '''
    z   = squeeze(z)
    h,w = shape(z)
    a   = np.abs(z)
    if vm is None: vm = numpy.max(a)
    if aspect is None: aspect = w/float(h)
    if onlyphase:
        rgb = complexHLArr2RGB(0.5*z/np.abs(z))
    else:
        rgb = complexHLArr2RGB(z*(0.9/vm))
    if previous is None:
        cla()
        img = imshow(rgb,cmap=None,aspect=aspect,interpolation=ip,extent=extent,origin=origin)
        draw()
        show()
        return img
    else:
        rgba = ones(shape(rgb)[:2]+(4,))
        rgba[:,:,:3] = rgb
        previous.set_data(rgba)
        draw()
        return previous

def animate_complex(z,vm=None,aspect='auto',ip='bicubic',
    extent=None,onlyphase=False,previous=None,origin='lower'):
    '''
    Like plot_complex except has an additional dimention for time
    '''
    p = None
    for frame in z:
        p=plot_complex(frame,vm,aspect,ip,extent,onlyphase,p,origin)

def good_colorbar(vmin,vmax,cmap,title='',ax=None,sideways=False,
    border=True,spacing=5,fontsize=12):
    '''
    Matplotlib's colorbar function is pretty bad. This is less bad.
    r'$\mathrm{\mu V}^2$'

    Parameters:
        vmin (number): min value for colormap
        vmax (number): mac value for colormap
        cmap (colormap): what colormap to use
        ax (axis): optional, defaults to plt.gca(). axis to which to add colorbar
        title (string): Units for colormap
        sideways (bool): Flips the axis label sideways
        spacing (int): distance from axis in pixels. defaults to 5
    Returns:
        axis: colorbar axis
    '''
    oldax = plt.gca() #remember previously active axis
    if ax==None: 
        ax=plt.gca()
    # WIDTH   = 0.05
    SPACING = pixels_to_xfigureunits(spacing,ax=ax)
    CWIDTH  = pixels_to_xfigureunits(15,ax=ax)
    # manually add colorbar axes 
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    # ax.set_position((x,y,w-WIDTH,h))
    bb = ax.get_position()
    right,bottom = bb.xmax,bb.ymax
    cax = plt.axes((right+SPACING,bottom-h,CWIDTH,h),facecolor='w',frameon=border)
    plt.sca(cax)
    plt.imshow(np.array([np.linspace(vmax,vmin,100)]).T,
        extent=(0,1,vmin,vmax),
        aspect='auto',
        cmap=cmap)
    nox()
    nicey()
    cax.yaxis.tick_right()
    if sideways:
        plt.text(
            xlim()[1]+pixels_to_xunits(5,ax=cax),
            np.mean(ylim()),
            title,
            fontsize=fontsize,
            rotation=0,
            horizontalalignment='left',
            verticalalignment  ='center')
    else:
        plt.ylabel(title,fontsize=fontsize)
    cax.yaxis.set_label_position("right")
    plt.sca(oldax) #restore previously active axis
    return cax

def complex_axis(scale):
    '''
    Draws a nice complex-plane axis with LaTeX Re, Im labels and everythinn
    '''
    xlim(-scale,scale)
    ylim(-scale,scale)
    nicexy()
    fudgey()
    ybartext(0,'$\Im(z)$','k','w',lw=1,color='k',outline=False)
    xbartext(0,'$\Re(z)$','k','w',lw=1,color='k',outline=False,horizontalalignment='right')
    noaxis()
    xlabel(u'μV',fontname='DejaVu Sans')
    ylabel(u'μV',fontname='DejaVu Sans')
    force_aspect()

def subfigurelabel(x,subplot_label_size=14,dx=20,dy=5):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    fontproperties = {
        'family':'Bitstream Vera Sans',
        'weight': 'bold',
        'size': subplot_label_size,
        'verticalalignment':'bottom',
        'horizontalalignment':'right'}
    text(xlim()[0]-pixels_to_xunits(dx),ylim()[1]+pixels_to_yunits(dy),x,**fontproperties)

def sigbar(x1,x2,y,pvalue,dy=5,LABELSIZE=10):
    '''
    draw a significance bar between position x1 and x2 at height y 
    
    Parameters
    ----------
    '''
    dy = pixels_to_yunits(dy)
    height = y+2*dy
    plot([x1,x1,x2,x2],[height-dy,height,height,height-dy],lw=0.5,color=BLACK)
    text(np.mean([x1,x2]),height+dy,shortscientific(pvalue),fontsize=LABELSIZE,horizontalalignment='center')

def savefigure(name):
    '''
    Saves figure as both SVG and PDF, prepending the current date
    in YYYYMMDD format
    
    Parameters
    ----------
    name : string
        file name to save as (sans extension)
    '''
    # strip user-supplied extension if present
    dirname  = os.path.dirname(name)
    if dirname=='': dirname='./'
    basename = os.path.basename(name)
    if basename.split('.')[-1].lower() in {'svg','pdf','png'}:
        basename = '.'.join(basename.split('.')[:-1])
    savefig(dirname + os.path.sep + today()+'_'+basename+'.svg',transparent=True,bbox_inches='tight')
    savefig(dirname + os.path.sep + today()+'_'+basename+'.pdf',transparent=True,bbox_inches='tight')
    savefig(dirname + os.path.sep + today()+'_'+basename+'.png',transparent=True,bbox_inches='tight')

def clean_y_range(ax=None,precision=1):
    '''
    Round down to a specified number of significant figures
    
    Parameters
    ----------
    
    '''
    if ax is None: ax=plt.gca()
    y1,y2 = ylim()
    precision = 10.0**precision
    _y1 = floor(y1*precision)/precision
    _y2 = ceil (y2*precision)/precision
    ylim(min(_y1,ylim()[0]),max(_y2,ylim()[1]))

def round_to_precision(x,precision=1):
    '''
    Round to a specified number of significant figures
    
    Parameters
    ----------
    x : scalar
        Number to round
    precision : positive integer, default=1
        Number of digits to keep
    
    Returns
    -------
    x : scalar
        Rounded number
    '''
    if x==0.0: return 0
    magnitude = np.abs(x)
    digits = np.ceil(np.log10(magnitude))
    factor = 10.0**(precision-digits)
    precision *= factor
    return np.round(x*precision)/precision

def ceil_to_precision(x,precision=1):
    '''
    Round up to a specified number of significant figures
    
    
    Parameters
    ----------
    x : scalar
        Number to round
    precision : positive integer, default=1
        Number of digits to keep
    
    Returns
    -------
    x : scalar
        Rounded number
    -------
    '''
    if x==0.0: return 0
    magnitude = np.abs(x)
    digits = np.ceil(np.log10(magnitude))
    factor = 10.0**(precision-digits)
    precision *= factor
    return np.ceil(x*precision)/precision

def floor_to_precision(x,precision=1):
    '''
    Round down to a specified number of significant figures
    
    Parameters
    ----------
    x : scalar
        Number to round
    precision : positive integer, default=1
        Number of digits to keep
    
    Returns
    -------
    x : scalar
        Rounded number
    '''
    if x==0.0: return 0
    magnitude = np.abs(x)
    digits = np.ceil(np.log10(magnitude))
    factor = 10.0**(precision-digits)
    precision *= factor
    return np.floor(x*precision)/precision

def expand_y_range(yvalues,ax=None,precision=1,pad=1.2):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax=plt.gca()
    yy = np.array(yvalues)
    m = np.mean(yy)
    yy = (yy-m)*pad+m
    y1 = np.min(yy)
    y2 = np.max(yy)
    precision = 10.0**precision
    _y1 = floor_to_precision(y1,precision)
    _y2 = ceil_to_precision(y2,precision)
    ylim(min(_y1,ylim()[0]),max(_y2,ylim()[1]))


def Gaussian2D_covellipse(M,C,N=60,**kwargs):
    '''
    xy = Gaussian2D_covellipse(M,C,N=60,**kwargs)

    Plot a covariance ellipse for 2D Gaussian with mean M and covariance C
    Ellipse is drawn at 1 standard deviation

    Parameters
    ----------
    M : tuple of (x,y) coordinates for the mean
    C : 2x2 np.array-like covariance matrix
    N : optional, number of points in ellipse (default 60)

    Returns
    -------
    xy : list of points in the ellipse
    '''
    circle = np.exp(1j*np.linspace(0,2*pi,N+1))
    xy = np.array([circle.real,circle.imag])
    xy = scipy.linalg.sqrtm(C).dot(xy)+M[:,None]
    plot(*xy,**kwargs);
    return xy

