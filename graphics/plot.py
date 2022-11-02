#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Plotting helper routines
'''
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
try: # python 2.x
    from itertools import izip, chain
except: # python 3
    from itertools import chain
    izip = zip

import os
import pickle
import scipy
import numpy
import scipy.optimize
import matplotlib.pyplot as plt

from scipy.io          import savemat, loadmat
from scipy.optimize    import leastsq
from scipy.signal      import butter,filtfilt,lfilter
from multiprocessing   import Process, Pipe, cpu_count, Pool

from neurotools.graphics.color import *
from neurotools.util.tools  import find
from neurotools.util.time   import today,now
from neurotools.util.string import shortscientific

from matplotlib.pyplot import *

try:
    import statsmodels
    import statsmodels.api as smapi
    import statsmodels.graphics as smgraphics
except:
    print('could not find statsmodels; some plotting functions missing')


def simpleaxis(ax=None):
    '''
    Only draw the bottom and left axis lines
    
    Other Parameters
    -----------------
    ax : maplotlib.Axis; default `plt.gca()`
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.autoscale(enable=True, axis='x', tight=True)
    
def rightaxis(ax=None):
    '''
    Only draw the bottom and right axis lines.
    Move y axis to the right.
    
    Parameters
    ----------
    ax : maplotlib.Axis; default `plt.gca()`
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.get_xaxis().tick_bottom()
    ax.autoscale(enable=True, axis='x', tight=True)

def simpleraxis(ax=None):
    '''
    Only draw the left y axis, nothing else
    
    Parameters
    ----------
    ax : maplotlib.Axis; default `plt.gca()`
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.autoscale(enable=True, axis='x', tight=True)
    
def simplerright(ax=None):
    '''
    Only draw the left y axis, nothing else
    
    Parameters
    ----------
    ax : maplotlib.Axis; default `plt.gca()`
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_right()
    ax.get_yaxis().set_label_position("right")
    ax.autoscale(enable=True, axis='x', tight=True)

def noaxis(ax=None):
    '''
    Hide all axes
    
    Parameters
    ----------
    ax : maplotlib.Axis; default `plt.gca()`
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def nicebp(bp,color='k',linewidth=.5):
    '''
    Improve the appearance of a box and whiskers plot.
    To be called on the object returned by the matplotlib boxplot function,
    with accompanying color information.
    
    Parameters
    ----------
    bp : point to boxplot object returned by matplotlib
    
    Other Parameters
    ----------------
    c: matplotlib.color; default `'k'`
        Color to set boxes to
    linewidth: positive float; default 0.5
        Width of whisker lines. 
    '''
    for kk in 'boxes whiskers fliers caps'.split():
        setp(bp[kk], color=color)
    setp(bp['whiskers'], linestyle='solid',linewidth=linewidth)
    setp(bp['caps'],     linestyle='solid',linewidth=linewidth)
    #setp(bp['caps'], color=(0,)*4)

def colored_boxplot(
    data,
    positions,
    color,
    filled      = True,
    notch       = False,
    showfliers  = False,
    lw          = 1,
    whis        = [5,95],
    bgcolor     = WHITE,
    mediancolor = None,
    **kwargs):
    '''
    Boxplot with nicely colored default style parameters
    
    Parameters
    ----------
    data: NPOINTS × NGROUPS np.float32
        Data sets to plot
    positions: NGROUPS iterable of numbers
        X positions of each data group
    color: matplotlib.color
        Color of boxplot   
    
    Other Parameters
    ----------------
    filled: boolean; default True
        Whether to fill boxes with color
    notch: boolean; default False
        Whether to inset a median notch
    showfliers: boolean; default False
        Whether to show outlies as scatter points
    lw: positive float; default 1.
        Width of whisker lines
    which: tuple; default (5,95)
        Percentile range for whiskers
    bgcolor: matplotlib.color; default WHITE
        Background color if `filled=False`
    mediancolor: matplotlib.color; default None
        Defaults to BLACK unless color is BLACK, in which
        case it defaults to WHITE.
    **kwargs:
        Additional arguments fowarded to `pyplot.boxplot()`
    '''
    if 'linewidth' in kwargs:
        lw = kwargs[linewidth]
    b = matplotlib.colors.to_hex(BLACK)
    if mediancolor is None:
        try:
            mediancolor = [BLACK if matplotlib.colors.to_hex(c)!=b else WHITE for c in color]
        except:
            mediancolor = BLACK if matplotlib.colors.to_hex(color)!=b else WHITE
    bp = plt.boxplot(data,
        positions    = positions,
        patch_artist = True,
        showfliers   = showfliers,
        notch        = notch,
        whis         = whis, 
        medianprops  = {'linewidth':lw,'color':mediancolor},
        whiskerprops = {'linewidth':lw,'color':color},
        flierprops   = {'linewidth':lw,'color':color},
        capprops     = {'linewidth':lw,'color':color},
        boxprops     = {'linewidth':lw,'color':color,
                        'facecolor':color if filled else bgcolor},
        **kwargs);
    return bp

def nicey(**kwargs):
    '''
    Mark only the min/max value of y axis
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ylim()[0]<0:
        plt.yticks([plt.ylim()[0],0,plt.ylim()[1]])
    else:
        plt.yticks([plt.ylim()[0],plt.ylim()[1]])
    fudgey(**kwargs)

def nicex(**kwargs):
    '''
    Mark only the min/max value of x axis
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if xlim()[0]<0:
        plt.xticks([plt.xlim()[0],0,plt.xlim()[1]])
    else:
        plt.xticks([plt.xlim()[0],plt.xlim()[1]])
    fudgex(**kwargs)

def nicexy(xby=None,yby=None,**kwargs):
    '''
    Mark only the min/max value of y/y axis. See `nicex` and `nicey`
    '''
    nicex(by=xby,**kwargs)
    nicey(by=yby,**kwargs)

def positivex():
    '''
    Sets the lower x limit to zero, and the upper limit to the largest
    positive value un the current xlimit. If the curent xlim() is
    negative, a value error is raised.
    '''
    top = np.max(xlim())
    if top<=0:
        raise ValueError('Current axis view lies within negative '+
            'numbers, cannot crop to a positive range')
    plt.xlim(0,top)
    nicex()

def positivey():
    '''
    Sets the lower y limit to zero, and the upper limit to the largest
    positive value un the current ylimit. If the curent ylim() is
    negative, a value error is raised.
    '''
    top = np.max(ylim())
    if top<=0:
        raise ValueError('Current axis view lies within negative '+
            'numbers, cannot crop to a positive range')
    plt.ylim(0,top)
    nicey()

def positivexy():
    '''
    Remove negative range from both x and y axes. See `positivex` and
    `positivey`
    '''
    positivex()
    positivey()

def xylim(a,b,ax=None):
    '''
    set x and y axis limits to the smae range
    
    Parameters
    ----------
    a : lower limit
    b : upper limit
    '''
    if ax==None: ax = plt.gca()
    ax.set_xlim(a,b)
    ax.set_ylim(a,b)


def noclip(ax=None):
    '''
    Turn of clipping
    '''
    if ax is None: 
        ax = plt.gca()
    for o in ax.findobj():
        o.set_clip_on(False)

def notick(ax=None,axis='both',which='both'):
    '''
    Hide axis ticks, but not their labels
    '''
    if ax is None:
        ax = plt.gca()
    ax.tick_params(axis, length=0, width=0, which=which)

def nox():
    '''
    Hide x-axis
    '''
    plt.xticks([])
    plt.xlabel('')

def noy():
    '''
    Hide y-axis
    '''
    plt.yticks([])
    plt.ylabel('')

def noxyaxes():
    '''
    Hide all aspects of x and y axes. See `nox`, `noy`, and `noaxis`
    '''
    nox()
    noy()
    noaxis()
    
def noxlabels():
    '''
    Hide x tick labels and x axis label
    '''
    plt.tick_params(axis='x',which='both',labelbottom=False)
    xlabel('')

def noylabels():
    '''
    Hide y tick labels and y axis label
    '''
    plt.tick_params(axis='y',which='both',labelbottom=False)    
    ylabel('')

def nolabels():
    '''
    Hide tick labels and axis labels
    '''
    noxlabels()
    noylabels()

def righty(ax=None):
    '''
    Move the y-axis to the right
    '''
    if ax==None: ax=plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

def unity(by=5,**kwargs):
    '''
    Set y-axis to unit interval
    '''
    ylim(0,1)
    nicey(by=by,**kwargs)

def unitx():
    '''
    Set x-axis to unit interval
    '''
    xlim(0,1)
    nicex()

def force_aspect(aspect=1,a=None):
    '''
    Parameters
    ----------
    aspect : aspect ratio
    '''
    if a is None: a = plt.gca()
    x1,x2=a.get_xlim()
    y1,y2=a.get_ylim()
    a.set_aspect(np.abs((x2-x1)/(y2-y1))/aspect)

def get_aspect(aspect=1,a=None):
    '''
    Returns
    ----------
    aspect : aspect ratio of current axis
    '''
    if a is None: a = plt.gca()
    x1,x2=a.get_xlim()
    y1,y2=a.get_ylim()
    return np.abs((x2-x1)/(y2-y1))

def match_image_aspect(im,ax=None):
    '''
    Keep upper-left corner of axis fixed, but
    rescale width and height to match dimensions
    of the given image. 

    Parameters
    ----------
    im: image instance to match
    '''
    if ax  is None: ax  = plt.gca()
    h,w = im.shape[:2]
    target_aspect = w/h
    x,y,w,h = get_bbox()
    y2 = y+h
    w = xfigureunits_to_pixels(w)
    h = yfigureunits_to_pixels(h)
    g = np.sqrt(target_aspect/(w/h))
    w3 = pixels_to_xfigureunits(w*g)
    h3 = pixels_to_yfigureunits(h/g)
    gca().set_position((x,y2-h3,w3,h3))

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
    dy = np.diff(ylim())[0]
    dx = np.diff(xlim())[0]
    return dx/float(w),dy/float(h)

def get_ax_pixel_ratio(ax=None,fig=None):
    '''
    Gets tha axis aspect ratio from pixel size
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    a,b = get_ax_pixel(ax,fig)
    return a/b

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
    dx = np.diff(xlim())[0]
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

def xfigureunits_to_pixels(n,ax=None,fig=None):
    '''
    Converts a measurement in figure-width units to units of
    x-axis pixels

    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w_pixels = fig.get_size_inches()[0]*fig.dpi
    return n*float(w_pixels)

def yfigureunits_to_pixels(n,ax=None,fig=None):
    '''
    Converts a measurement in figure-height units to units of
    y-axis pixels
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    h_pixels = fig.get_size_inches()[1]*fig.dpi
    return n*float(h_pixels)
    
    

# aliases
px2x = pixels_to_xunits
px2y = pixels_to_yunits

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

def get_bbox(ax=None):
    '''
    Get bounding box of currenta axis
    '''
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    return x,y,w,h

def nudge_axis_y_pixels(dy,ax=None):
    '''
    moves axis dx pixels.
    Direction of dx may depend on axis orientation.
    Does not change axis height.

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
    resize axis by dy pixels.
    Direction of dx may depends on axis orientation.
    Does not change the baseline position of axis.
    
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
    This does not change the height of the axis

    Parameters
    ----------
    dy : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,ax)
    ax.set_position((x,y+dy,w,h))

def nudge_axis_x(dx,ax=None):
    '''
    This does not change the width of the axis

    Parameters
    ----------
    dx : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,ax)
    ax.set_position((x+dx,y,w,h))

def expand_axis_x(dx,ax=None):
    '''
    Expands the width of the x axis

    Parameters
    ----------
    dx : number
        Amount (in pixels) to adjust x axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,ax)
    ax.set_position((x,y,w+dx,h))
    
def expand_axis_y(dy,ax=None):
    '''
    Adjusts the axis height, keeping the lower y-limit the same

    Parameters
    ----------
    dy : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,ax)
    ax.set_position((x,y,w,h+dy))

def nudge_axis_baseline(dy,ax=None):
    '''
    Moves bottom limit of axis, keeping top limit the same

    Parameters
    ----------
    dy : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,ax)
    ax.set_position((x,y+dy,w,h-dy))

def nudge_axis_left(dx,ax=None):
    '''
    Moves the left x-axis limit, keeping the right limit intact. 
    This changes the width of the plot.

    Parameters
    ----------
    dx : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,ax)
    ax.set_position((x+dx,y,w-dx,h))

def zoombox(ax1,ax2,xspan1=None,xspan2=None,draw_left=True,draw_right=True,lw=1,color='k'):
    '''
    Parameters
    ----------
    ax1:
    ax2:
    xspan1:None
    xspan2:None
    draw_left:True
    draw_right:True
    lw:1
    color:'k'
    '''
    # need to do this to get the plot to ... update correctly
    draw()
    fig = plt.gcf()

    if xspan1==None:
        xspan1 = ax1.get_xlim()
    if xspan2==None:
        xspan2 = ax2.get_xlim()
    transFigure = fig.transFigure.inverted()
    if draw_left:
        coord1 = transFigure.transform(ax1.transData.transform([xspan1[0],ax1.get_ylim()[1]]))
        coord2 = transFigure.transform(ax2.transData.transform([xspan2[0],ax2.get_ylim()[0]]))
        line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                       transform=fig.transFigure,lw=lw,color=color)
        fig.lines.append(line)
    if draw_right:
        coord1 = transFigure.transform(ax1.transData.transform([xspan1[1],ax1.get_ylim()[1]]))
        coord2 = transFigure.transform(ax2.transData.transform([xspan2[1],ax2.get_ylim()[0]]))
        line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                       transform=fig.transFigure,lw=lw,color=color)
        fig.lines.append(line)

def fudgex(by=None,ax=None,doshow=False):
    '''
    Adjust x label spacing in pixels

    Parameters
    ----------
    by : number of pixels
    axis : axis object to change; defaults to current axis
    dishow : boolean; if true, calls plt.show()
    '''
    if ax is None: ax=plt.gca()
    if by is None:
        if min(ax.get_xlim())<0 and max(ax.get_xlim())>0:
            by = 0
        else:
            by = 10
    ax.xaxis.labelpad = -by
    plt.draw()
    if doshow:
        plt.show()

def fudgey(by=None,ax=None,doshow=False):
    '''
    Adjust y label spacing in pixels

    Parameters
    ----------
    by : number of pixels
    axis : axis object to change; defaults to current axis
    dishow : boolean; if true, calls plt.show()
    '''
    if ax is None: ax=plt.gca()
    if by is None:
        if min(ax.get_ylim())<0 and max(ax.get_ylim())>0:
            by = 5
        else:
            by = 13
    ax.yaxis.labelpad = -by
    plt.draw()
    if doshow:
        plt.show()

def fudgexy(by=10,ax=None):
    '''
    Adjust x and y label spacing in pixels

    Parameters
    ----------
    by : number of pixels
    axis : axis object to change; defaults to current axis
    dishow : boolean; if true, calls plt.show()
    '''
    fudgex(by,ax)
    fudgey(by,ax)

def shade_edges(edges,color=(0.5,0.5,0.5,0.5)):
    '''
    Edges of the form (start,stop)
    Shades regions of graph defined by "edges"

    Parameters
    ----------
    edges
    color
    '''
    a,b = ylim()
    c,d = xlim()
    for x1,x2 in zip(*edges):
        print(x1,x2)
        fill_between([x1,x2],[a,a],[b,b],color=color,lw=0)
    ylim(a,b)
    xlim(c,d)

def ybartext(x,t,c1,c2,**kwargs):
    '''
    Parameters
    ----------
    y:
    t:
    c1:
    c2:
    
    Other Parameters
    ----------------
    **kwargs: keyword arguments forwarded to plot() and text()
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
    y:
    t:
    c1:
    c2:

    Other Parameters
    ----------------
    **kwargs: keyword arguments forwarded to plot() and text()
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
    Better defaults for the plot legend.

    Other Parameters
    ----------------
    *args: arguments forwarded to legend()
    **kwargs: keyword arguments forwarded to legend()
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

def right_legend(*args,**kwargs):
    '''
    Legend outside the plot to the right.

    Other Parameters
    ----------------
    *args: arguments forwarded to legend()
    **kwargs: keyword arguments forwarded to legend()
    '''
    defaults = {
        'loc':'center left',
        'bbox_to_anchor':(1,0.5),
        }
    defaults.update(kwargs)
    lg = legend(*args,**defaults)
    lg.get_frame().set_linewidth(0.0)
    return lg

def left_legend(*args,**kwargs):
    '''
    Legend outside the plot to the left.

    Other Parameters
    ----------------
    *args: arguments forwarded to legend()
    **kwargs: keyword arguments forwarded to legend()
    '''
    x = -0.2
    if 'fudge' in kwargs:
        x = kwargs['fudge']
        del kwargs['fudge'] 
    defaults = {
        'loc':'center right',
        'bbox_to_anchor':(x,0.5),
        }
    defaults.update(kwargs)
    lg = legend(*args,**defaults)
    lg.get_frame().set_linewidth(0.0)
    return lg

def base_legend(*args,fudge=-0.1,**kwargs):
    '''
    Legend outside the plot on the base.

    Other Parameters
    ----------------
    fudge: padding between legend and axis, default -0.1
    '''
    y = fudge
    defaults = {
        'loc':'upper center',
        'bbox_to_anchor':(0.5,y),
        }
    defaults.update(kwargs)
    lg = legend(*args,**defaults)
    lg.get_frame().set_linewidth(0.0)
    return lg

def rangeto(rangefun,data):
    '''
    Parameters
    ----------
    rangefun:
    data:
    
    '''
    rangefun(np.min(data),np.max(data))

def rangeover(data):
    '''
    Parameters
    ----------
    data:
    '''
    return np.min(data),np.max(data)

def cleartop(x):
    '''
    Parameters
    ----------
    x:
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
    cwt  = np.squeeze(cwt)
    nf,N = np.shape(cwt)
    pwr  = np.abs(cwt)
    fest = ff[np.argmax(pwr,0)]
    plt.cla()
    imshow(pwr,aspect=aspect,extent=(0,N,ff[-1],ff[0]),vmin=vmin,vmax=vmax,interpolation=interpolation,cmap=cm)
    plt.xlim(0,N)
    plt.ylim(ff[0],ff[-1])
    try:
        plt.tight_layout()
    except:
        print('tight_layout missing, how old is your python? seriously')
    if dodraw:
        plt.draw()
        plt.show()

def complex_axis(scale):
    '''
    Draws a nice complex-plane axis with LaTeX Re, Im labels and everything
    
    Parameters
    ----------
    
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
    
    '''
    cwt = np.squeeze(cwt)
    nf,N = np.shape(cwt)
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
    
    '''
    cwt  = np.squeeze(cwt)
    nf,N = np.shape(cwt)
    pwr    = np.abs(cwt)
    fest   = ff[np.argmax(pwr,0)]
    plt.clf()
    plt.subplot(211)
    plt.imshow(pwr,aspect=aspect,extent=(0,N,ff[-1],ff[0]),
        vmin=vmin,vmax=vmax,cmap=cm,interpolation=interpolation)
    plt.xlim(0,N)
    plt.ylim(ff[0],ff[-1])
    plt.subplot(212)
    plt.imshow(angle(cwt),aspect=aspect,extent=(0,N,ff[-1],ff[0]),
        vmin=vmin,vmax=vmax,cmap=medhue,interpolation=interpolation)
    plt.xlim(0,N)
    plt.ylim(ff[0],ff[-1])
    try:
        plt.tight_layout()
    except:
        print('tight_layout missing, you should update')
    plt.draw()
    plt.show()

def domask(*args):
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
    '''

    Parameters
    ----------
    '''
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width,height, fontsize, trans):
        '''
        Parameters
        ----------
        
        '''
        center = xdescent + 0.5 * (width-height),ydescent
        p = mpatches.Rectangle(xy=center,width=height,height=height, angle=0.0)
        self.update_prop(p,orig_handle,legend)
        p.set_transform(trans)
        return [p]

def plot_complex(z,vm=None,aspect='auto',ip='bicubic',
    extent=None,
    onlyphase=False,
    previous=None,
    origin='lower'):
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
    z   = np.squeeze(z)
    h,w = np.shape(z)
    a   = np.abs(z)
    if vm     is None: vm     = numpy.max(a)
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
    Like plot_complex except has an additional dimension for time

    Parameters
    ----------
    '''
    p = None
    for frame in z:
        p=plot_complex(frame,vm,aspect,ip,extent,onlyphase,p,origin)


def good_colorbar(vmin=None,
    vmax=None,
    cmap=None,
    title='',
    ax=None,
    sideways=False,
    border=True,
    spacing=5,
    width=15,
    labelpad=10,
    fontsize=10,
    vscale=1.0,
    va='c'):
    '''
    Matplotlib's colorbar function is pretty bad. This is less bad.
    r'$\mathrm{\mu V}^2$'

    Parameters:
        vmin     (number)  : min value for colormap
        vmax     (number)  : mac value for colormap
        cmap     (colormap): what colormap to use
        title    (string)  : Units for colormap
        ax       (axis)    : optional, defaults to plt.gca(). axis to which to add colorbar
        sideways (bool)    : Flips the axis label sideways
        border   (bool)    : Draw border around colormap box? 
        spacing  (number)  : distance from axis in pixels. defaults to 5
        width    (number)  : width of colorbar in pixels. defaults to 15
        labelpad (number)  : padding between colorbar and title in pixels, defaults to 10
        fontsize (number)  : label font size, defaults to 12
        vscale   (float)   : height adjustment relative to parent axis, defaults to 1.0
        va       (str)     : vertical alignment; "bottom" ('b'), "center" ('c'), or "top" ('t')
    Returns:
        axis: colorbar axis
    '''
    if type(vmin)==matplotlib.image.AxesImage:
        img  = vmin
        cmap = img.get_cmap()
        vmin = img.get_clim()[0]
        vmax = img.get_clim()[1]
        ax   = img.axes
    oldax = plt.gca() #remember previously active axis
    if ax is None: ax=plt.gca()
    SPACING = pixels_to_xfigureunits(spacing,ax=ax)
    CWIDTH  = pixels_to_xfigureunits(width,ax=ax)
    # manually add colorbar axes 
    bb = ax.get_position()
    x,y,w,h,r,b = bb.xmin,bb.ymin,bb.width,bb.height,bb.xmax,bb.ymax
    y0 = {
        'b':lambda:b-h,
        'c':lambda:b-(h+h*vscale)/2,
        't':lambda:b-h*vscale
    }[va[0]]()
    cax = plt.axes((r+SPACING,y0,CWIDTH,h*vscale),frameon=True)
    plt.sca(cax)
    plt.imshow(np.array([np.linspace(vmax,vmin,100)]).T,
        extent=(0,1,vmin,vmax),
        aspect='auto',
        origin='upper',
        cmap=cmap)
    nox()
    nicey()
    cax.yaxis.tick_right()
    if sideways:
        plt.text(
            xlim()[1]+pixels_to_xunits(labelpad,ax=cax),
            np.mean(ylim()),
            title,
            fontsize=fontsize,
            rotation=0,
            horizontalalignment='left',
            verticalalignment  ='center')
    else:
        plt.text(
            xlim()[1]+pixels_to_xunits(labelpad,ax=cax),
            np.mean(ylim()),
            title,
            fontsize=fontsize,
            rotation=90,
            horizontalalignment='left',
            verticalalignment  ='center')
    # Hide ticks
    #noaxis()
    cax.tick_params('both', length=0, width=0, which='major')
    cax.yaxis.set_label_position("right")
    cax.yaxis.tick_right()
    plt.sca(oldax) #restore previously active axis
    return cax

def complex_axis(scale):
    '''
    Draws a nice complex-plane axis with LaTeX Re, Im labels and everything

    Parameters
    ----------
    '''
    xlim(-scale,scale)
    ylim(-scale,scale)
    nicexy()
    fudgey()
    ybartext(0,'$\Im(z)$','k','w',lw=1,color='k',outline=False)
    xbartext(0,'$\Re(z)$','k','w',lw=1,color='k',outline=False,ha='right')
    noaxis()
    xlabel(u'μV',fontname='DejaVu Sans')
    ylabel(u'μV',fontname='DejaVu Sans')
    force_aspect()

def subfigurelabel(x,fontsize=10,dx=39,dy=7,ax=None,bold=True,**kwargs):
    '''
    Parameters
    ----------
    x : label
    '''
    if ax is None: ax = plt.gca()
    fontproperties = {
        'fontsize':fontsize,
        'family':'Bitstream Vera Sans',
        'weight': 'bold' if bold else 'normal',
        'va':'bottom',
        'ha':'left'}
    fontproperties.update(kwargs)
    text(xlim()[0]-pixels_to_xunits(dx),ylim()[1]+pixels_to_yunits(dy),x,
        **fontproperties)

def sigbar(x1,x2,y,pvalue=None,dy=5,padding=1,fontsize=10,color=BLACK,**kwargs):
    '''
    draw a significance bar between position x1 and x2 at height y 
    
    Parameters
    ----------
    x1: float
        X position of left of brace
    x2: float
        X position of right of brace
    y: float
        Y position to start brace
    
    Other Parameters
    ----------------
    dy: float; default 5
        Brace height in pixels
    padding: float; default 1
        Padding between brace and label, in pixels
    fontsize: float; default 10
        Label font size
    color: matplotlib.color; default BLACK
        Brace color
    **kwargs:
        Forwarded to the `plot()` command that draws the
        brace.
    '''
    dy = pixels_to_yunits(dy)
    height = y+2*dy
    if not 'lw' in kwargs:
        kwargs['lw']=0.5
    plot([x1,x1,x2,x2],[height-dy,height,height,height-dy],
        color=color,clip_on=False,**kwargs)
    if not pvalue is None:
        if not type(pvalue) is str:
            pvalue = shortscientific(pvalue)
        text(np.mean([x1,x2]),height+dy*padding,pvalue,
            fontsize=fontsize,horizontalalignment='center')

def hsigbar(y1,y2,x,pvalue=None,dx=5,padding=1,fontsize=10,color=BLACK,**kwargs):
    '''
    Draw a significance bar between position y1 and y2 at 
    horizontal position x.
    
    Parameters
    ----------
    y1 : float
    y2 : float
    x  : float
    
    Other Parameters
    ----------------
    dx: float; default 5
        Brace width in pixels
    padding: float; default 1
        Padding between brace and label, in pixels
    fontsize: float; default 10
        Label font size
    color: matplotlib.color; default BLACK
        Brace color
    **kwargs:
        Forwarded to the `plot()` command that draws the
        brace.
    '''
    dx = pixels_to_xunits(dx)
    w = x+2*dx
    if not 'lw' in kwargs:
        kwargs['lw']=0.5
    plot([w-dx,w,w,w-dx],[y1,y1,y2,y2],
        color=color,clip_on=True,**kwargs)
    if not pvalue is None:
        if not type(pvalue) is str:
            pvalue = shortscientific(pvalue)
        text(w+dx*padding,np.mean([y1,y2]),pvalue,
            fontsize=fontsize,ha='left',va='center')

def savefigure(name,stamp=True,**kwargs):
    '''
    Saves figure as both SVG and PDF, prepending the current date-ti,me
    in YYYYMMDD_HHMMSS format
    
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
    if not 'dpi' in kwargs:
        kwargs['dpi']=600
    prefix = now()+'_'+basename if stamp else basename
    savefig(dirname + os.path.sep+prefix+'.svg',
        transparent=True,bbox_inches='tight',pad_inches=0,**kwargs)
    savefig(dirname + os.path.sep+prefix+'.pdf',
        transparent=True,bbox_inches='tight',pad_inches=0,**kwargs)
    savefig(dirname + os.path.sep+prefix+'.png',
        transparent=True,bbox_inches='tight',pad_inches=0,**kwargs)

def clean_y_range(ax=None,precision=1):
    '''
    Round down to a specified number of significant figures
    
    Other Parameters
    ----------------
    ax: axis
    precision: int
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
    digits    = np.ceil(np.log10(magnitude))
    factor    = 10.0**(precision-digits)
    return np.round(x*factor)/factor

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
    yvalues
    
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

def stderrplot(m,v,color='k',alpha=0.1,smooth=None,lw=1.5,
    filled=True,label=None,stdwidth=1.96):
    '''
    Plot mean±1.96*σ
    
    Parameters
    ----------
    m : mean
    v : variance
    
    Other Parameters
    ----------------
    color : 
        Plot color
    alpha : 
        Shaded confidence alpha color blending value
    smooth : int
        Number of samples over which to smooth the variance
    '''
    plot(m, color = color,lw=lw,label=label)
    e = np.sqrt(v)*stdwidth
    if not smooth is None and smooth>0:
        e = neurotools.signal.box_filter(e,smooth)
        m = neurotools.signal.box_filter(m,smooth)
    if filled:
        c = mpl.colors.colorConverter.to_rgb(color)+(alpha ,)
        fill_between(np.arange(len(m)),m-e,m+e,lw=0,color=c)
    else:
        plot(m-e,':',lw=lw*0.5,color=color)
        plot(m+e,':',lw=lw*0.5,color=color)    

def yscalebar(ycenter,yheight,label,x=None,color='k',fontsize=9,ax=None,side='left'):
    '''
    Add vertical scale bar to plot
    
    Parameters
    ----------
    Returns
    -------
    Other Parameters
    ----------------
    ax : axis, if None (default), uses the current axis.
    '''
    yspan = [ycenter-yheight/2.0,ycenter+yheight/2.0]
    if ax is None:
        ax = plt.gca()
    plt.draw() # enforce packing of geometry
    if x is None:
        x = -pixels_to_xunits(5)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    plt.plot([x,x],yspan,
        color='k',
        lw=1,
        clip_on=False)
    if side=='left':
        plt.text(x-pixels_to_xunits(2),np.mean(yspan),label,
            rotation=90,
            fontsize=fontsize,
            horizontalalignment='right',
            verticalalignment='center',
            clip_on=False)
    else:
        plt.text(x+pixels_to_xunits(2),np.mean(yspan),label,
            rotation=90,
            fontsize=fontsize,
            horizontalalignment='left',
            verticalalignment='center',
            clip_on=False)
        
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)

def xscalebar(xcenter,xlength,label,y=None,color='k',fontsize=9,ax=None):
    '''
    Add horizontal scale bar to plot
    
    Parameters
    ----------
    xcenter: float
        Horizontal center of the scale bar
    xlength: float
        How wide the scale bar is
    '''
    xspan = [xcenter-xlength/2.0,xcenter+xlength/2.0]
    if ax is None:
        ax = plt.gca()
    plt.draw() # enforce packing of geometry
    if y is None:
        y = -pixels_to_yunits(5)
    yl = ax.get_ylim()
    xl = ax.get_xlim()
    plt.plot(xspan,[y,y],
        color=color,
        lw=1,
        clip_on=False)
    plt.text(np.mean(xspan),y-pixels_to_yunits(5),label,
        fontsize=fontsize,
        horizontalalignment='center',
        verticalalignment='top',
        color=color,
        clip_on=False)
    ax.set_ylim(*yl)
    ax.set_xlim(*xl)
        
def addspikes(Y,lw=0.2,color='k'):
    '''
    Add vertical lines where Y>0
    Parameters
    ----------
    Returns
    -------
    '''
    for t in find(Y>0): 
        axvline(t,lw=lw,color=color)
        
def unit_crosshairs(draw_ellipse=True,draw_cross=True):
    '''
    Parameters
    ----------
    Returns
    -------
    '''
    lines  = []
    if draw_ellipse:
        circle = np.exp(1j*np.linspace(0,2*np.pi,361))
        lines += list(circle)
    if draw_cross:
        line1  = np.linspace(-1,1,50)
        line2  = 1j*line1
        lines += [np.nan]+list(line1)+[np.nan]+list(line2)
    lines = np.array(lines)
    return np.array([lines.real,lines.imag])

def covariance_crosshairs(S,p=0.8,draw_ellipse=True,draw_cross=True):
    '''
    For 2D Gaussians these are the confidence intervals
    p   | sigma
    90  : 4.605
    95  : 5.991
    97.5: 7.378
    99  : 9.210
    99.5: 10.597

    Parameters
    ----------
    S: 2D covariance matrix
    p: fraction of data ellipse should enclose
    '''
    sigma = np.sqrt(scipy.stats.chi2.isf(1-p,df=2))
    e,v = scipy.linalg.eigh(S)
    lines = unit_crosshairs(draw_ellipse,draw_cross)*sigma
    lines *= (e**0.5)[:,None]
    return scipy.linalg.pinv(v).dot(lines)

from matplotlib.patches import Arc, RegularPolygon
def draw_circle(radius,centX,centY,angle_,theta2_,
                arrowsize=1,
                ax=None,
                cap_start=1,
                cap_end=1,
                **kwargs):
    '''
    Parameters
    ----------
    ax : axis, if None (default), uses the current axis.
    '''
    if ax is None: ax = plt.gca()
    arc = Arc([centX,centY],radius,radius,angle=angle_*180/np.pi,
          theta1=0,theta2=theta2_*180/np.pi,capstyle='round',linestyle='-',**kwargs)
    ax.add_patch(arc)
    if cap_end:
        endX=centX+(radius/2)*np.cos(theta2_+angle_)
        endY=centY+(radius/2)*np.sin(theta2_+angle_)
        ax.add_patch(RegularPolygon((endX,endY),3,arrowsize,angle_+theta2_,**kwargs))
    if cap_start:
        endX=centX+(radius/2)*np.cos(angle_)
        endY=centY+(radius/2)*np.sin(angle_)
        ax.add_patch(RegularPolygon((endX,endY),3,arrowsize,angle_+np.pi,**kwargs))

def simple_arrow(x1,y1,x2,y2,ax=None,s=5,color='k',lw=1.5,**kwargs):
    '''
    Connect two points with a triangular error.
    
    Parameters
    ----------
    x1: float
    y1: float
    x2: float
    y2: float
    
    Other Parameters
    ----------------
    ax : axis, if None (default), uses the current axis.
    s: float; passed as the `headlength` and `headwidth` arrow property.
    width: float; line width
    color: matplotlib color
    '''
    if ax is None: ax = plt.gca()
    ax.annotate(None, 
                xy=(x2,y2), 
                xytext=(x1,y1), 
                xycoords='data',
                textcoords='data',
                arrowprops=dict(shrink=0,width=lw,lw=0,
                    color=color,headwidth=s,headlength=s),
                **kwargs)

def inhibition_arrow(x1,y1,x2,y2,ax=None,width=0.5,color='k'):
    '''
    Connect two points with a `T` (inhibition; braking) arrow.
    
    Parameters
    ----------
    x1: float
    y1: float
    x2: float
    y2: float

    Other Parameters
    ----------------
    ax : axis; if None (default), uses the current axis.
    width: float; line width
    color: matplotlib color
    '''
    if ax is None: ax = plt.gca()
    ax.annotate(None,
                xy=(x2,y2), 
                xytext=(x1,y1), 
                xycoords='data',
                textcoords='data',
                arrowprops={ 'arrowstyle':        
                    matplotlib.patches.ArrowStyle.BracketB(
                        widthB=width,lengthB=0),
                    'color':color
                })

def figurebox(color=(0.6,0.6,0.6)):
    '''
    Draw a colored border around the edge of a figure.
    '''
    # new clear axis overlay with 0-1 limits
    from matplotlib import pyplot, lines
    ax2 = pyplot.axes([0,0,1,1],facecolor=(1,1,1,0))# axisbg=(1,1,1,0))
    x,y = np.array([[0,0,1,1,0], [0,1,1,0,0]])
    line = lines.Line2D(x, y, lw=1, color=color)
    ax2.add_line(line)
    plt.xticks([]); plt.yticks([]); noxyaxes()

def more_xticks(ax=None):
    '''
    Add more ticks to the x axis

    Other Parameters
    ----------------
    ax : axis, if None (default), uses the current axis.
    '''
    if ax is None: ax = plt.gca()
    xticks     = ax.get_xticks()
    xl         = ax.get_xlim()
    xmin,xmax  = xl
    mintick    = np.min(xticks)
    maxtick    = np.max(xticks)
    nticks     = len(xticks)
    new_xticks = np.linspace(mintick,maxtick,nticks*2-1)
    spacing    = np.mean(np.diff(sorted(xticks)))
    before     = mintick - spacing/2
    if before>=xmin and before<=xmax:
        new_xticks = np.concatenate([[before],new_xticks])
    after      = maxtick + spacing/2
    if after>=xmin and after<=xmax:
        new_xticks = np.concatenate([new_xticks,[after]])
    ax.set_xticks(new_xticks)

def more_yticks(ax=None):
    '''
    Add more ticks to the y axis

    Other Parameters
    ----------------
    ax : axis, if None (default), uses the current axis.
    '''
    if ax is None:
        ax = plt.gca()
    yticks     = ax.get_yticks()
    yl         = ax.get_ylim()
    ymin,ymax  = yl
    mintick    = np.min(yticks)
    maxtick    = np.max(yticks)
    nticks     = len(yticks)
    new_yticks = np.linspace(mintick,maxtick,nticks*2-1)
    spacing    = np.mean(np.diff(sorted(yticks)))
    before     = mintick - spacing/2
    if before>=ymin and before<=ymax:
        new_yticks = np.concatenate([[before],new_yticks])
    after      = maxtick + spacing/2
    if after>=ymin and after<=ymax:
        new_yticks = np.concatenate([new_yticks,[after]])
    ax.set_yticks(new_yticks)

def border_width(lw=0.4,ax=None):
    '''
    Adjust width of axis border

    Parameters
    ----------
    lw : line width of axis borders to use

    Other Parameters
    ----------------
    ax : axis, if None (default), uses the current axis.
    '''
    if ax is None: ax = gca()
    [i.set_linewidth(lw) for i in ax.spines.values()]


def broken_step(x,y,eps=1e-5,*args,**kwargs):
    '''
    Draws a step plot but does not connect
    adjacent levels with vertical lines

    Parameters
    ----------
    x: horizontal position of steps
    y: height of steps 

    Other Parameters
    ----------------
    eps: step size above which to break; default is 1e-5
    **args: arguments forwarded to plot()
    **kwargs: keyword arguments forwarded to plot()
    '''
    x = np.float32(x).ravel()
    y = np.float32(y).ravel()
    skip = np.where(np.abs(np.diff(y))>eps)[0]+1
    k = len(skip)
    n = np.ones(k)*np.NaN
    # Insert np.NaN to break apart segments
    x = np.insert(x,skip,n)
    y = np.insert(y,skip,n)
    # Update location of breaks to match new locations after insertion
    skip = skip + np.arange(k)
    # Get the x location, and y value from left and right, of each skip
    xmid = (x[skip-1]+x[skip+1])*.5
    yneg = y[skip-1]
    ypos = y[skip+1]
    # Before every break, insert a point interpolating 1/2 sample forward
    x = np.insert(x,skip-1,xmid)
    y = np.insert(y,skip-1,yneg)
    # Update location of breaks to match new locations after insertion
    skip = skip + np.arange(k) + 1
    # After every break, insert a point interpolating 1/2 sample prior
    x = np.insert(x,skip+1,xmid)
    y = np.insert(y,skip+1,ypos)
    plt.plot(x,y,*args,**kwargs)


def label(x="",y="",t=""):
    """
    Convenience function for setting x label, y label, and
    title in one command.

    Parameters
    ----------
    x: string, optional; x-axis label
    y: string, optional; y-axis label
    t: string, optional; title
    """
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(t)

def flathist(x):
    '''
    '''
    x = np.array(x)
    s = np.shape(x)
    x = np.ravel(x)
    x = scipy.stats.rankdata(x)
    return np.reshape(x,s)

def barcompare(x,y,bins=20,mode='p50',markersize=7,lw=1.5,**kwargs):
    '''
    Bar plot of Y as a function of X, summarized in bins of X
    '''
    if type(bins) is int:
        skip  = len(x)//bins
        bins  = array(sorted(x))[::skip]
        bins[-1] = np.max(y)+1e-9
    nbins = len(bins)-1
    means,stds,sems = [],[],[]
    p50 = []
    Δe = (bins[1:]+bins[:-1])*0.5
    for i in range(nbins):
        ok = (x>=bins[i])&(x<bins[i+1])
        n  = np.sum(ok)
        v  = np.var(y[ok])
        m  = np.mean(y[ok])
        means.append(m)
        stds.append(np.sqrt(v))
        sems.append(np.sqrt(v/n)*1.96)
        try:
            p50.append(m-np.percentile(y[ok],[25,75]))
        except:
            e = 0.6742864804798947*np.sqrt(v)*n/(n-1)
            p50.append([e,e])
    μ  = np.array(means)
    σ  = np.array(stds)
    dμ = np.array(sems)
    errs = {'sem':dμ,'p95':1.96*σ,'p50':abs(np.array(p50)).T}[mode]
    plt.errorbar(Δe, μ, errs,
                 fmt='.',
                 markersize=markersize,
                 lw=lw,
                 capsize=0,
                 zorder=np.inf,**kwargs)
    
def shellmean(x,y,bins=20):
    '''
    Get mean and standard deviation of Y based on histogram
    bins of X
    μ, σ, dμ, Δe = shellmean(x,y,bins=20)
    '''
    if type(bins) is int:
        skip  = len(x)//bins
        bins  = array(sorted(x))[::skip]
        bins[-1] = np.max(y)+1e-9
    nbins = len(bins)-1
    means,stds,sems = [],[],[]
    Δe = (bins[1:]+bins[:-1])*0.5
    for i in range(nbins):
        ok = (x>=bins[i])&(x<bins[i+1])
        n  = np.sum(ok)+1
        v  = np.nanvar(y[ok])
        if not isfinite(v):
            v = 0
        m = np.nanmean(y[ok])
        if not isfinite(m):
            m = 0
        if len(y[ok])<1:
            m = NaN
        means.append(m)
        stds.append(np.sqrt(v))
        sems.append(np.sqrt(v/n)*1.96)
    μ  = np.array(means)
    σ  = np.array(stds)
    dμ = np.array(sems)
    return μ, σ, dμ, Δe

def trendline(x,y,ax=None,color=RUST):
    '''
    Parameters
    ----------
    x : x points
    y : y points
    ax : figure axis for plotting, if None uses plt.gca()
    '''
    if ax is None:
        ax = plt.gca()
    m,b = np.polyfit(x,y,1)
    xl = np.array(ax.get_xlim())
    plt.plot(xl,xl*m+b,label='offset = %0.2f\nslope = %0.2f'%(b,m),color=color)
    ax.set_xlim(*xl)
    plt.legend(edgecolor=(1,)*4)

def shellplot(x,y,z,SHELLS,label='',vmin=None,vmax=None,ax=None):
    '''
    Averages X and Y based on bins of Z
    '''
    Xμ, σ, dμ, Δe = shellmean(z,x,bins=SHELLS)
    Yμ, σ, dμ, Δe = shellmean(z,y,bins=SHELLS)
    ok = isfinite(Xμ)
    Xμ = Xμ[ok]
    Yμ = Yμ[ok]
    Δe = Δe[ok]
    if ax is None:
        smallplot()
    x = Xμ
    y = Yμ
    ns = len(x)
    plt.scatter(x,y,c=Δe,lw=0,s=16,vmin=vmin,vmax=vmax)
    cbar = plt.colorbar()
    simpleaxis()
    #cbar.set_ticks(arange(vmin,,2))
    cbar.ax.set_ylabel(label)
    trendline(x,y)
    return x,y,Δe

def arrow_between(A,B,size=None):
    '''
    Draw an arrow between two matplotlib axis instances
    '''
    draw()
    fig = plt.gcf()

    position = A.get_position().transformed(fig.transFigure)
    ax0,ay0,ax1,ay1 = position.x0,position.y0,position.x1,position.y1
    position = B.get_position().transformed(fig.transFigure)
    bx0,by0,bx1,by1 = position.x0,position.y0,position.x1,position.y1

    # arrow outline
    cx  = array([0,1.5,1.5,3,1.5,1.5,0])
    cy  = array([0,0,-0.5,0.5,1.5,1,1])
    cx -= (np.max(cx)-np.min(cx))/2
    cy -= (np.max(cy)-np.min(cy))/2
    cwidth = np.max(cx)-np.min(cx)

    horizontal = vertical = None
    if   max(ax0,ax1)<min(bx0,bx1): horizontal = -1 # axis A is to the left of axis B
    elif max(bx0,bx1)<min(ax0,ax1): horizontal =  1 # axis A is to the right of axis B
    elif max(ay0,ay1)<min(by0,by1): vertical   = -1 # axis A is above B
    elif max(by0,by1)<min(ay0,ay1): vertical   =  1 # axis A is below B
    assert not (horizontal is None     and vertical is None    )
    assert not (horizontal is not None and vertical is not None)

    if horizontal is not None:
        x0 = max(*((ax0,ax1) if horizontal==-1 else (bx0,bx1)))
        x1 = min(*((bx0,bx1) if horizontal==-1 else (ax0,ax1)))
        span     = x1 - x0
        pad      = 0.1 * span
        width    = span - 2*pad
        scale = width/cwidth if size is None else size
        px = -horizontal*cx*scale + (x0+x1)/2
        py = cy*scale + (ay0+ay1+by0+by1)/4
        polygon = Polygon(array([px,py]).T,facecolor=BLACK)#,transform=tt)
        fig.patches.append(polygon)

    if vertical is not None:
        y0 = max(*((ay0,ay1) if vertical==-1 else (by0,by1)))
        y1 = min(*((by0,by1) if vertical==-1 else (ay0,ay1)))
        span     = y1 - y0
        pad      = 0.1 * span
        width    = span - 2*pad
        scale = width/cwidth if size is None else size
        px = -vertical*cx*scale + (y0+y1)/2
        py = cy*scale + (ax0+ax1+bx0+bx1)/4
        polygon = Polygon(array([py,px]).T,facecolor=BLACK)#,transform=tt)
        fig.patches.append(polygon)
        
def splitz(z,thr=1e-9):
    '''
    Split a 1D complex signal into real and imaginary parts, setting
    components that are zero in either to np.NaN. This lets us plot components
    separately without overlap (see `plotz()`).
    '''
    z   = np.complex64(z)
    r,i = np.float32(np.real(z)),np.float32(np.imag(z))
    r[abs(r)<thr]=np.NaN
    i[abs(i)<thr]=np.NaN
    return r,i

def plotz(x,z,thr=1e-9,**k):
    '''
    Plot a 1D complex signal, drawing the imaginary component as a dashed line.
    '''
    r,i = splitz(z,thr=thr)
    anyr = np.any(~np.isnan(r))
    anyi = np.any(~np.isnan(i))
    if anyr:
        l = None
        if 'label' in k:
            l = k['label']
            if anyi:
                l += r' $(\Re)$'
        plot(x,r,**{**k,'linestyle':'-','label':l})
    if anyi:
        l=k['label']+r'$(\Im)$' if 'label' in k else None
        plot(x,i,**{**k,'linestyle':':','label':l})


__SAVE_LIMITS_PRIVATE_STORAGE__ = None
def save_limits():
    '''
    Stash the current ((x0,x1),(y0,y1) axis limits in
    `__SAVE_LIMITS_PRIVATE_STORAGE__`.
    These can be restored later via `restore_limits()`
    '''
    global __SAVE_LIMITS_PRIVATE_STORAGE__
    __SAVE_LIMITS_PRIVATE_STORAGE__ = (plt.xlim(),plt.ylim())


def restore_limits():
    '''
    Restore the ((x0,x1),(y0,y1) limits stored in 
    `__SAVE_LIMITS_PRIVATE_STORAGE__`
    '''
    global __SAVE_LIMITS_PRIVATE_STORAGE__
    xl,yl = __SAVE_LIMITS_PRIVATE_STORAGE__
    plt.xlim(*xl)
    plt.ylim(*yl)
    

def mock_legend(names,colors,s=40,marker='s'):
    '''
    For a list of (labels, colors), generate some 
    square scatter points outside the axis limits
    with the given labels, so that the `legend()` call
    will populate a list of labelled, colored squares.
    
    Parameters
    ----------
    labels: list of str
        List of labels to create
    colors: list of matplotlib.color
        List ofl label colors, same length as labels
    '''
    save_limits()
    x0 = xlim()[0]-100
    y0 = ylim()[0]-100
    for n,c in zip(names,colors):
        scatter(x0,y0,s=s,color=c,marker=marker,label=n)
    restore_limits()
    
def xtickpad(pad=0,ax=None,which='both'):
    '''
    Adjust padding of xticks to axis
    
    Parameters
    ----------
    pad: positive float; default 0
        Distance between axis and ticks
        
    Other Parameters
    ----------------
    ax: matplotlib.axis or None; default None
        If `None`, uses `pyplot.gca()`
    which: str in {'major','minor','both'}; default 'both'
        Which set of ticks to apply to
    '''
    if ax is None:
        ax = plt.gca()
    ax.tick_params(axis='x', which=which, pad=pad)

def ytickpad(pad=0,ax=None,which='both'):
    '''
    Adjust padding of yticks to axis
    
    Parameters
    ----------
    pad: positive float; default 0
        Distance between axis and ticks
        
    Other Parameters
    ----------------
    ax: matplotlib.axis or None; default None
        If `None`, uses `pyplot.gca()`
    which: str in {'major','minor','both'}; default 'both'
        Which set of ticks to apply to
    '''
    if ax is None:
        ax = plt.gca()
    ax.tick_params(axis='y', which=which, pad=pad)

def xticklen(l=0, w=0,ax=None,which='both'):
    '''
    Set length and width of x ticks.
    
    Parameters
    ----------
    l: positive float; default 0
        Length of ticks
    w: positive float; default 0
        Width of ticks
        
    Other Parameters
    ----------------
    ax: matplotlib.axis or None; default None
        If `None`, uses `pyplot.gca()`
    which: str in {'major','minor','both'}; default 'both'
        Which set of ticks to apply to
    '''
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_tick_params(length=l, width=w, which=which)

def yticklen(l=0, w=0,ax=None,which='both'):
    '''
    Set length and width of y ticks.
    
    Parameters
    ----------
    l: positive float; default 0
        Length of ticks
    w: positive float; default 0
        Width of ticks
        
    Other Parameters
    ----------------
    ax: matplotlib.axis or None; default None
        If `None`, uses `pyplot.gca()`
    which: str in {'major','minor','both'}; default 'both'
        Which set of ticks to apply to
    '''
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_tick_params(length=l, width=w, which=which)

def xin(ax=None,which='both'):
    '''
    Make x ticks point inward
    
    Other Parameters
    ----------------
    ax: matplotlib.axis or None; default None
        If `None`, uses `pyplot.gca()`
    which: str in {'major','minor','both'}; default 'both'
        Which set of ticks to apply to
    '''
    if ax is None:
        ax = plt.gca()
    ax.tick_params(axis='x',which=which,direction="in")
    
def yin(ax=None,which='both'):
    '''
    Make y ticks point inward
    
    Other Parameters
    ----------------
    ax: matplotlib.axis or None; default None
        If `None`, uses `pyplot.gca()`
    which: str in {'major','minor','both'}; default 'both'
        Which set of ticks to apply to
    '''
    if ax is None:
        ax = plt.gca()
    ax.tick_params(axis='y',which=which,direction="in")

def lighten(color,amount):
    '''
    Lighten matplotlib color by amount (0=do nothing).
    Alpha channel is discarded
    
    Parameters
    ----------
    color: matplotlib.color
        Color to lighten
    amount: float in [0,1]
        Amount to lighten by. 0: do nothing, 1: make white.
    
    Returns
    -------
    rgb: 
        Lightened color
    '''
    rgb = np.float32(mpl.colors.to_rgb(color))
    rgb = 1.-(1.-rgb)*(1-amount)
    return rgb

def darken(color,amount):
    '''
    Lighten matplotlib color by amount (0=do nothing).
    Alpha channel is discarded
    
    Parameters
    ----------
    color: matplotlib.color
        Color to lighten
    amount: float in [0,1]
        Amount to lighten by. 0: do nothing, 1: make white.
    
    Returns
    -------
    rgb: 
        Lightened color
    '''
    rgb = np.float32(mpl.colors.to_rgb(color))
    rgb = rgb*(1-amount)
    return rgb
        
def axvstripe(edges,colors,fade=0.0,**kwargs):
    '''
    Shade vertical spans of edges in alternating bands.
    
    Parameters
    ----------
    edges: list of numbers
        x coordinates of edges of shaded bands
    colors: color or list of colors
        If a single color, will alternated colored/white.
        If a list of colors, will rotate within list
    alpha: positive float ∈[0,1]; default 0
        Amount of white to mix into the colors
    '''
    edges  = np.float32(edges).ravel()
    nbands = len(edges)-1
    
    try: 
        colors = [lighten(colors,fade),None]
    except:
        colors = [lighten(c,fade) for c in colors]
    NCOLORS = len(colors)
    
    kw = {'linewidth':0,'edgecolor':(0,)*4,**kwargs}
    
    for i in range(nbands):
        x0 = edges[i]
        x1 = edges[i+1]
        c  = colors[i%NCOLORS]
        if not c is None:
            plt.axvspan(x0,x1,facecolor=c,**kw)

def widths_to_edges(widths,startat=0):
    '''
    Convert a list of widths into a list of edges
    delimiting consecutive bands of the given width
    
    Parameters
    ----------
    widths: list of numbers
        Width of each band
    startat: number, default 0
        Starting position of bands
    '''
    edges = np.cumsum(np.concatenate([[0],widths]))
    edges = edges + float(startat)
    return edges

def centers(edges):
    '''
    Get center of histogram bins given as a list of edges.
    
    Parameters
    ----------
    edges: list of numbers
        Edges of histogram bins
        
    Returns
    -------
    list of numbers
        Center of histogram bins
    '''
    edges = np.float32(edges)
    return (edges[1:]+edges[:-1])*0.5

def widths_to_centers(widths,startat=0):
    '''
    Get centers of a consecutive collection of histogram
    widths. 
    
    Parameters
    ----------
    widths: list of numbers
        Width of each band
    startat: number, default 0
        Starting position of bands
    '''
    edges = widths_to_edges(widths,startat=startat)
    return centers(edges)

def axvbands(widths,colors=BLACK,fade=0.8,startat=0,**kwargs):
    '''
    Wrapper for `axvstripe` that accepts band widths 
    rather than edges.
    
    Parameters
    ----------
    widths: list of numbers
        Width of each band
    colors: color or list of colors
        If a single color, will alternated colored/white.
        If a list of colors, will rotate within list
    alpha: positive float ∈[0,1]; default 0
        Amount of white to mix into the colors
    startat: number, default 0
        Starting position of bands
    '''
    axvstripe(
        widths_to_edges(widths,startat=startat),
        colors=colors,fade=fade,**kwargs)


def zerohline(color='k',lw=None):
    '''
    Draw horizontal line at zero matching axis style.
    '''
    if lw is None:
        lw = matplotlib.rcParams['axes.linewidth']
    plt.axhline(0,color=color,lw=lw)


def zerovline(color='k',lw=None):
    '''
    Draw vertical line at zero matching axis style.
    '''
    if lw is None:
        lw = matplotlib.rcParams['axes.linewidth']
    plt.axvline(0,color=color,lw=lw)


import scipy.stats
import neurotools.stats.pvalues as pvalues
def boxplot_significance(
    a1,
    positionsa,
    b1=None,
    positionsb=None,
    fdr=0.05,
    dy=5,
    fontsize=6,
    label_pvalue=True,
    significance_mark='∗'):
    '''
    Perform Wilcoxon tests on a pair of box-plot sets
    and add significance brackets.
        
    This corrects for multiple comparisons using the
    Benjamini-Hochberg procedure, using either the 
    variance for positive dependence or no dependence,
    whichever is more conservative. 
        
    Parameters
    ----------
    a1: NGROUPS×NPOINTS np.float32
        Condition A
    positionsa: NGROUPS iterable of numbers
        X positions of each of group A
        
    Other Parameters
    ----------------
    b1: NGROUPS×NPOINTS np.float32; Default None
        Condition B
    positionsb: NGROUPS iterable of numbers; Default None
        X positions of each of group B
    fdr: float in (0,1); default 0.05
        Desired false discovery rate for 
        Benjamini Hochberg correction
    dy: positive number; default 5
        Padding, in pixels, between box and p-value annotation
    fontsize: postiive float; default 6
        Font size of p-value, if shown
    label_pvalue: boolean; default True
        Only for single-population tests.
        Whether to draw corrected p-value for significant 
        boxes.
    significance_mark: str; default '∗'
        Only for single-population tests.
        Marker to use to note significant boxes.    
        
    Returns
    ------
    pvalues: np.float32
        List of corrected pvalues for each box or comparison
        between a pair of boxes
    is_significant: np.bool
        List of booleans indicating whether given box or
        pair of boxes was signfiicant at the specified
        falst discovery rate threshold after correcting
        for multiple comparisons.
    '''
    def test(*args):
        try: 
            return scipy.stats.wilcoxon(*args).pvalue
        except ValueError:
            return np.NaN
    
    if b1 is None:
        # One dataset: test if different from zero
        pv = np.float32([test(ai) for ai in a1]).ravel()
        pv2, reject = pvalues.correct_pvalues(pv,alpha=fdr,verbose=False)
        hy = np.array([np.percentile(ai,95) for ai in a1]).ravel()
        for ip,ir,x,y in zip(pv2,reject,positionsa,hy):
            if ip<fdr:
                s = significance_mark
                if label_pvalue:
                    s = shortscientific(ip)+'\n'+s
                text(x,y+px2y(dy),s,fontsize=fontsize,ha='center')
    else:
        # Two datasets: test if different
        # Perform tests
        pv = np.float32([test(ai,bi) for ai,bi, in zip(a1,b1)]).ravel()
        pv2, reject = pvalues.correct_pvalues(pv,alpha=fdr,verbose=False)
        # Bracket heights
        hy = np.array([[
            max(np.percentile(ai,95),np.percentile(bi,95)) 
            for ai,bi in zip(aa,bb)] 
            for aa,bb in [(a1,b1)]]).T.ravel()
        for ip,ir,x1,x2,y in zip(pv2,reject,positionsa,positionsb,hy):
            if ip<fdr:
                sigbar(x1,x2,y,pvalue=ip,dy=dy,fontsize=fontsize)
    return pv2, reject


def pikeplot(x,y,**kwargs):
    '''
    Plot timeseries as verical lines dropped to y=0.
    Keyword arguments are forwarded to `pyplot.plot()`.
    
    Parameters
    ----------
    x: 1D np.array
        X Location of bars
    y: 1D np.array
        Y position of bars
    '''
    y = np.array(y)
    y = np.array([y*0,y,np.NaN*y]).T.ravel()
    x = np.array([x,x,np.NaN*x]).T.ravel()
    plt.plot(x,y,**kwargs)