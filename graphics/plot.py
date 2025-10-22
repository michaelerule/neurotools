#!/usr/bin/python3
# -*- coding: UTF-8 -*-
'''
Plotting helper routines
'''
import os, sys, pickle
import numpy, scipy
import scipy.optimize
import scipy.stats
import matplotlib        as mpl
import matplotlib.pyplot as plt

from scipy.io                  import savemat, loadmat
from scipy.optimize            import leastsq
from scipy.signal              import butter,filtfilt,lfilter
from multiprocessing           import Process, Pipe, cpu_count, Pool
from matplotlib.pyplot         import *
from matplotlib.patches        import Polygon
from matplotlib.collections    import PatchCollection

from neurotools.util.array     import find, centers
from neurotools.util.array     import widths_to_edges
from neurotools.util.array     import widths_to_centers
from neurotools.util.time      import today, now
from neurotools.util.string    import shortscientific
from neurotools.graphics.color import *

try:
    import statsmodels
    import statsmodels.api as smapi
    import statsmodels.graphics as smgraphics
except:
    print('no statsmodels; some plotting functions missing')


############################################################
# Shorthand for common axes options

def simpleaxis(ax=None):
    '''
    Only draw the bottom and left axis lines
    
    Other Parameters
    -----------------
    ax: maplotlib.Axis; default ``plt.gca()``
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
    ax: maplotlib.Axis; default ``plt.gca()``
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
    ax: maplotlib.Axis; default ``plt.gca()``
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.autoscale(enable=True, axis='x', tight=True)

def simplestaxis(ax=None):
    '''
    Parameters
    ----------
    ax: maplotlib.Axis; default ``plt.gca()``
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.autoscale(enable=True, axis='x', tight=True)
    
def simplerright(ax=None):
    '''
    Only draw the left y axis, nothing else
    
    Parameters
    ----------
    ax: maplotlib.Axis; default ``plt.gca()``
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
    ax: maplotlib.Axis; default ``plt.gca()``
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.autoscale(enable=True, axis='x', tight=True)

def nicey(**kwargs):
    '''
    Mark only the min/max value of y axis
    
    Other Parameters
    ----------------
    **kwargs:dict
        Keyword arguments are forwarded to ``fudgey(**kwargs)``
    
    Returns
    -------
    '''
    if plt.ylim()[0]<0:
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
    if plt.xlim()[0]<0:
        plt.xticks([plt.xlim()[0],0,plt.xlim()[1]])
    else:
        plt.xticks([plt.xlim()[0],plt.xlim()[1]])
    fudgex(**kwargs)

def nicexy(xby=None,yby=None,**kwargs):
    '''
    Mark only the min/max value of y/y axis. 
    See ``nicex`` and ``nicey``
    '''
    nicex(by=xby,**kwargs)
    nicey(by=yby,**kwargs)

def roundx(to=10):
    '''
    Round x axis limits to an integer multiple of ``to``.
    '''
    x0 = int(np.floor(plt.xlim()[0]/to))*to
    x1 = int(np.ceil(plt.xlim()[1]/to))*to
    plt.xlim(x0,x1)

def roundy(to=10):
    '''
    Round y axis limits to an integer multiple of ``to``.
    '''
    y0 = int(np.floor(plt.ylim()[0]/to))*to
    y1 = int(np.ceil(plt.ylim()[1]/to))*to
    plt.ylim(y0,y1)

def positivex():
    '''
    Sets the lower x limit to zero, and the upper limit to 
    the largest positive value un the current xlimit. 
    If the curent plt.xlim() is negative, 
    a ``ValueError`` is raised.
    '''
    top = np.max(plt.xlim())
    if top<=0:
        raise ValueError(
            'Current axis view lies within negative '
            'numbers, cannot crop to a positive range')
    plt.xlim(0,top)
    nicex()

def positivey():
    '''
    Sets the lower y limit to zero, and the upper limit to 
    the largest positive value un the current ylimit. 
    If the curent plt.ylim() is negative, a ValueError is raised.
    '''
    top = np.max(plt.ylim())
    if top<=0:
        raise ValueError(
            'Current axis view lies within negative '
            'numbers, cannot crop to a positive range')
    plt.ylim(0,top)
    nicey()

def positivexy():
    '''
    Remove negative range from both x and y axes. See ``positivex`` and
    ``positivey``
    '''
    positivex()
    positivey()

def xylim(a,b,ax=None):
    '''
    set x and y axis limits to the same range
    
    Parameters
    ----------
    a: lower limit
    b: upper limit
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
    Hide all aspects of x and y axes. See ``nox``, ``noy``, and ``noaxis``
    '''
    nox()
    noy()
    noaxis()
    plt.gca().set_frame_on(False)
    
def noxlabels():
    '''
    Hide x tick labels and x axis label
    '''
    plt.tick_params(axis='x',which='both',labelbottom=False)
    plt.xlabel('')

def noylabels():
    '''
    Hide y tick labels and y axis label
    '''
    plt.tick_params(axis='y',which='both',labelbottom=False)    
    plt.ylabel('')

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
    plt.ylim(0,1)
    nicey(by=by,**kwargs)

def unitx():
    '''
    Set x-axis to unit interval
    '''
    plt.xlim(0,1)
    nicex()














############################################################
# Aspect and dimension control

def fsize(f=None):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    if f is None: f=plt.gcf()
    return f.get_size_inches()

def force_aspect(aspect=1,a=None):
    '''
    Parameters
    ----------
    aspect: aspect ratio
    
    Other Parameters
    ----------------
    a: matplotlib.Axis, default None
        If None, uses gca()
    '''
    if a is None: a = plt.gca()
    x1,x2=a.get_xlim()
    y1,y2=a.get_ylim()
    a.set_aspect(np.abs((x2-x1)/(y2-y1))/aspect)

def square_compare(hi=1,lo=0,**kwargs):
    '''
    Force an axis to be square and draw a diagonal 
    line for comparing equality of `x` and `y` values.
    '''
    if lo>hi: lo,hi=hi,lo
    simpleaxis()
    plot([lo,hi],[lo,hi],**(dict(color='k',lw=.6)|kwargs)) 
    plt.xlim(lo,hi)
    plt.ylim(lo,hi) 
    force_aspect()

def get_aspect(aspect=1,a=None):
    '''
    Other Parameters
    ----------------
    a: matplotlib.Axis, default None
        If None, uses gca()
    
    Returns
    -------
    aspect: aspect ratio of current axis
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
    
    Other Parameters
    ----------------
    ax: matplotlib.Axis, default None
        If None, uses gca()
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
    Set both `x` and `y` axis to span ``[0,1]``
    
    Other Parameters
    ----------------
    a: matplotlib.Axis, default None
        If None, uses gca()
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
    arraymap: 2D np.array
    
    Returns
    -------
    adjustmap: 
    '''
    nrow,ncol   = np.shape(arraymap)
    adjustedmap = np.array(arraymap)
    available   = sorted([*{*
        [x for x in np.ravel(arraymap) if x>0]
    }])
    for i,ch in enumerate(available):
        adjustedmap[arraymap==ch]=i
    return adjustedmap

def get_ax_size(ax=None,fig=None):
    '''
    Gets tha axis size in figure-relative units
    
    Other Parameters
    ----------------
    ax: matplotlib.Axis, default None
        If None, uses gca()
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    width: float
    height: float
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
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
    
    Other Parameters
    ----------------
    ax: matplotlib.Axis, default None
        If None, uses gca()
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    width: float
    height: float
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    # w/h in pixels
    w,h = get_ax_size()
    # one px in axis units is the axis span div no. pix
    dy = np.diff(plt.ylim())[0]
    dx = np.diff(plt.xlim())[0]
    return dx/float(w),dy/float(h)


def get_ax_pixel_ratio(ax=None,fig=None):
    '''
    Gets tha axis aspect ratio from pixel size
    
    Other Parameters
    ----------------
    ax: matplotlib.Axis, default None
        If None, uses gca()
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    aspect_ratio: float
    '''
    a,b = get_ax_pixel(ax,fig)
    return a/b


def pixels_to_xunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current x-axis
    scale
    
    Parameters
    ----------
    
    Other Parameters
    ----------------
    ax: matplotlib.Axis, default None
        If None, uses gca()
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    :float
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
    n: number
        y position in pixels
    
    Other Parameters
    ----------------
    ax: matplotlib.Axis, default None
        If None, uses gca()
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    :float
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
    n: number
        x position in pixes
    
    Other Parameters
    ----------------
    ax: matplotlib.Axis, default None
        If None, uses gca()
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    :float
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dx = np.diff(plt.xlim())[0]
    return n*float(w)/dx


def pixels_to_yunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the 
    current y-axis scale.
    
    Parameters
    ----------
    n: number
        number of pixels
    
    Other Parameters
    ----------------
    ax: matplotlib.Axis, default None
        If None, uses gca()
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    :float
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dy = np.diff(plt.ylim())[0]
    return n*dy/float(h)


def pixels_to_xfigureunits(n,fig=None):
    '''
    Converts a measurement in pixels to units of the current
    figure width scale.
    
    Parameters
    ----------
    n: number
        x coordinate in pixels.
    
    Other Parameters
    ----------------
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    :float
    '''
    if fig is None: fig = plt.gcf()
    w_pixels = fig.get_size_inches()[0]*fig.dpi
    return n/float(w_pixels)


# aliases
px2x = pixels_to_xunits
px2y = pixels_to_yunits

def pixels_to_yfigureunits(n,fig=None):
    '''
    Converts a measurement in pixels to units of the current
    figure height scale.
    
    Parameters
    ----------
    n: number
        y coordinate in pixels.
    
    Other Parameters
    ----------------
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    :float
    '''
    if fig is None: fig = plt.gcf()
    h_pixels = fig.get_size_inches()[1]*fig.dpi
    return n/float(h_pixels)


def xfigureunits_to_pixels(n,ax=None,fig=None):
    '''
    Converts a measurement in figure-width units to units of
    x-axis pixels

    Parameters
    ----------
    n: number
        x coordinate in figure-width units
    
    Other Parameters
    ----------------
    ax: matplotlib.Axis, default None
        If None, uses gca()
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    :float
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w_pixels = fig.get_size_inches()[0]*fig.dpi
    return n*float(w_pixels)

def yfigureunits_to_pixels(n,ax=None,fig=None):
    '''
    Converts a measurement in figure-height units to units 
    of y-axis pixels
    
    Parameters
    ----------
    n: number
        y coordinate in figure units
    
    Other Parameters
    ----------------
    ax: matplotlib.Axis, default None
        If None, uses gca()
    fig: matplotlib.Figure, default None
        If None, uses gcf()
    
    Returns
    -------
    :float
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    h_pixels = fig.get_size_inches()[1]*fig.dpi
    return n*float(h_pixels)

def adjust_ylabel_space(n,ax=None):
    '''
    
    Parameters
    ----------
    n: float 
        Desired value for ``ax.yaxis.labelpad``
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses gca()
    '''
    if ax is None: ax=plt.gca()
    ax.yaxis.labelpad = n

def adjust_xlabel_space(n,ax=None):
    '''
    
    Parameters
    ----------
    n: float 
        Desired value for ``ax.xaxis.labelpad``
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses gca()
    '''
    if ax is None: ax=plt.gca()
    ax.xaxis.labelpad = n

def get_bbox(ax=None):
    '''
    Get bounding box of currenta axis

    Parameters
    ----------
    ax: axis, default None
        If None, uses gca()
    
    Returns
    -------
    x:float
    y:float
    w:float
    h:float
    '''
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    return x,y,w,h

def nudge_axis_y_pixels(dy,ax=None,fig=None):
    '''
    Moves axis ``dx`` pixels.
    Direction of ``dx`` may depend on axis orientation.
    Does not change axis height.

    Parameters
    ----------
    dy: number
        Amount (in pixels) to adjust by
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses ``gca()``
    fig: figure, default NOne
        If None, uses ``gcf()``
    '''
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = -pixels_to_yfigureunits(float(dy),fig)
    ax.set_position((x,y-dy,w,h))

def adjust_axis_height_pixels(dy,ax=None,fig=None):
    '''
    resize axis by dy pixels.
    Direction of dx may depends on axis orientation.
    Does not change the baseline position of axis.
    
    Parameters
    ----------
    dy: number
        Amount (in pixels) to adjust by
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses ``gca()``
    fig: figure, default NOne
        If None, uses ``gcf()``
    '''
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    ax.set_position((x,y,w,h-pixels_to_yfigureunits(float(dy),fig)))

def nudge_axis_y(dy,ax=None,fig=None):
    '''
    This does not change the height of the axis

    Parameters
    ----------
    dy: number
        Amount (in pixels) to adjust by
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses ``gca()``
    fig: figure, default NOne
        If None, uses ``gcf()``
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,fig)
    ax.set_position((x,y+dy,w,h))

def nudge_axis_up(dy,ax=None):
    nudge_axis_y(dy,ax)

def nudge_axis_down(dy,ax=None):
    nudge_axis_y(-dy,ax)

def nudge_axis_x(dx,ax=None,fig=None):
    '''
    This does not change the width of the axis.

    Parameters
    ----------
    dx: number
        Amount (in pixels) to adjust axis
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses ``gca()``
    fig: figure, default NOne
        If None, uses ``gcf()``
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,fig)
    ax.set_position((x+dx,y,w,h))

def expand_axis_x(dx,ax=None,fig=None):
    '''
    Expands the width of the x axis

    Parameters
    ----------
    dx: number
        Amount (in pixels) to adjust x axis
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses ``gca()``
    fig: figure, default NOne
        If None, uses ``gcf()``
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,fig)
    ax.set_position((x,y,w+dx,h))
    
def expand_axis_y(dy,ax=None,fig=None):
    '''
    Adjusts the axis height, keeping the lower y-limit the 
    same.

    Parameters
    ----------
    dy: number
        Amount (in pixels) to adjust axis
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses ``gca()``
    fig: figure, default NOne
        If None, uses ``gcf()``
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,fig)
    ax.set_position((x,y,w,h+dy))

def nudge_axis_baseline(dy,ax=None,fig=None):
    '''
    Moves bottom limit of axis, keeping top limit the same.
    This will change the height of the y axis.

    Parameters
    ----------
    dy: number
        Amount (in pixels) to adjust axis
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses ``gca()``
    fig: figure, default NOne
        If None, uses ``gcf()``
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,fig)
    ax.set_position((x,y+dy,w,h-dy))

def nudge_axis_top(dy,ax=None,fig=None):
    '''
    Moves top limit of axis, keeping bottom limit the same.
    This will change the height of the y axis.

    Parameters
    ----------
    dy: number
        Amount (in pixels) to adjust axis
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses ``gca()``
    fig: figure, default NOne
        If None, uses ``gcf()``
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,fig)
    ax.set_position((x,y,w,h+dy))

def nudge_axis_left(dx,ax=None,fig=None):
    '''
    Moves the left x-axis limit, keeping the right limit 
    intact. This changes the width of the plot.

    Parameters
    ----------
    dx: number
        Amount (in pixels) to adjust axis
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses ``gca()``
    fig: figure, default NOne
        If None, uses ``gcf()``
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,fig)
    ax.set_position((x+dx,y,w-dx,h))

def nudge_axis_right(dx,ax=None,fig=None):
    '''
    Moves the right x-axis limit, keeping the left limit 
    intact. This changes the width of the plot.

    Parameters
    ----------
    dx: number
        Amount (in pixels) to adjust axis
        
    Other Parameters
    ----------------
    ax: axis, default None
        If None, uses ``gca()``
    fig: figure, default NOne
        If None, uses ``gcf()``
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,fig)
    ax.set_position((x,y,w+dx,h))
    
def expand_axis_outward(dx,dy=None):
    '''
    Expand all edges of axis outward by ``dx`` pixels.
    '''
    if dy is None: dy = dx
    ax = gca()
    fig = gcf()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,fig)
    dy = pixels_to_yfigureunits(dy,fig)
    ax.set_position((x-dx,y-dy,w+2*dx,h+2*dy))
    
def fudgex(by=None,ax=None,doshow=False):
    '''
    Adjust x label spacing in pixels

    Parameters
    ----------
    by: number of pixels
    axis: axis object to change; defaults to current axis
    dishow: boolean; if true, calls plt.show()
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
    by: number of pixels
    axis: axis object to change; defaults to current axis
    dishow: boolean; if true, calls plt.show()
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
    by: number of pixels
    axis: axis object to change; defaults to current axis
    dishow: boolean; if true, calls plt.show()
    '''
    fudgex(by,ax)
    fudgey(by,ax)

def vtwin(ax1,ax2):
    '''
    Assuming that ``ax2`` is directly below ``ax1``,
    move ``ax2`` up to directly abut the base of ``ax1``.
    (Used for making horizontal paired-histogram plots). 
    
    Parameters
    ----------
    ax1: axis 1
    ax2: axis 2
    '''
    bb = ax1.get_position()
    x1,y1,w1,h1 = bb.xmin,bb.ymin,bb.width,bb.height
    bb = ax2.get_position()
    x2,y2,w2,h2 = bb.xmin,bb.ymin,bb.width,bb.height
    dy = y1-(y2+h2)
    ax2.set_position((x1,y2 + dy,w1,h2))
    ax2.invert_yaxis()
    
    
    
    
    
    
    
    
    
    
    


def zoombox(
    ax1,ax2,
    xspan1=None,xspan2=None,
    draw_left=True,draw_right=True,
    lw=1,color='k'):
    '''
    Draw lines on figure connecting two axes, to indicate
    that one is a "zoomed in" version of the other. 
    
    The axes should be placed atop one another with the
    "zoomed in" axis below for this to work.
    
    Parameters
    ----------
    ax1: axis
        Axis to zoom in to
    ax2: axis
        Axis reflecting zoomed-in portion
    xspan1: tuple; default None
        ``(x_start, x_stop)`` span to connect on the 
        first axis.
    xspan2: tuple; default None
        ``(x_start, x_stop)`` span to connect on the 
        second axis.
    draw_left: boolean; default True
    draw_right: boolean; default True
    lw: number; default 1
    color: matplotlib color; default 'k'
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
    
def shade_edges(edges,color=(0.5,0.5,0.5,0.5),**kwargs):
    '''
    Edges of the form (start,stop)
    Shades regions of graph defined by "edges"

    Parameters
    ----------
    edges
    color
    '''
    a,b = plt.ylim()
    c,d = plt.xlim()
    for x1,x2 in zip(*edges):
        fill_between([x1,x2],[a,a],[b,b],color=color,lw=0,**kwargs)
    plt.ylim(a,b)
    plt.xlim(c,d)

def ybartext(
    x,t,
    c1='k',c2='w',
    outline=False,
    fontsize=12,
    **kwargs):
    '''
    Draws an ``axvline()`` spanning the *current*
    y-axis limits (``plt.ylim()``), with an appropriately
    spaced (and optionally outlined) label at the top
    left. 
    
    Parameters
    ----------
    y: number
        Y position 
    t: str
    c1: matplotlib color; default 'k'
        Text color.
    c2: matplotlib color; default 'w'
        Color of text outline, if ``outline=True``.
    outline: boolean; default False
        Whether to outline the text.
    
    Other Parameters
    ----------------
    **kwargs: dict
        keyword arguments forwarded to 
        ``plot()`` and ``text()``.
    '''
    a,b = plt.ylim()
    plot([x,x],[a,b],**(dict(lw=0.8,color='k')|kwargs))
    plt.ylim(a,b)
    dx,dy = get_ax_pixel()
    # Generate outline by drawing copies in the background
    # There must be a better way?
    if outline:
        for ix in arange(-2,3)*dx:
            for iy in arange(-2,3)*dy:
                text(ix+x,iy+plt.ylim()[1]-dy*4,t,
                    rotation=90,
                    color=c2,
                    ha='right',
                    va='top',
                    fontsize=fontsize)
    # Add text label
    text(
        x,
        plt.ylim()[1]-dy*2,
        t,
        rotation=90,
        color=c1,
        fontsize=fontsize,
        ha='right',
        va='top')
    # Restore y limits
    plt.ylim(a,b)

def xbartext(
    y,t,
    c1='k',c2='w',
    outline=False,
    fontsize=12,
    **kwargs):
    '''
    Draws an ``axhline()`` spanning the *current*
    x-axis limits (``plt.xlim()``), with an appropriately
    spacedd (and optionally outlined) label at the top
    left. 
    
    Parameters
    ----------
    x: number
        X position 
    t: str
    c1: matplotlib color; default 'k'
        Text color.
    c2: matplotlib color; default 'w'
        Color of text outline, if ``outline=True``.
    outline: boolean; default False
        Whether to outline the text.
    
    Other Parameters
    ----------------
    **kwargs: dict
        keyword arguments forwarded to 
        ``plot()`` and ``text()``.
    '''
    a,b = plt.xlim()
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
    plt.xlim(a,b)
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
    plt.xlim(a,b)

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

def right_legend(*args,fudge=0.0,**kwargs):
    '''
    Legend outside the plot to the right.

    Other Parameters
    ----------------
    *args: arguments forwarded to legend()
    **kwargs: keyword arguments forwarded to legend()
    '''
    defaults = {
        'loc':'center left',
        'bbox_to_anchor':(1+fudge,0.5),
        }
    defaults.update(kwargs)
    lg = plt.legend(*args,**defaults)
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
    lg = plt.legend(*args,**defaults)
    lg.get_frame().set_linewidth(0.0)
    return lg

def base_legend(*args,fudge=-0.1,**kwargs):
    '''
    Legend outside the plot on the base.

    Other Parameters
    ----------------
    fudge: padding between legend and axis, default -0.1
    '''
    lg = plt.legend(*args,**{**{
        'loc':'upper center',
        'bbox_to_anchor':(0.5,0.0+fudge),
        },**kwargs})
    lg.get_frame().set_linewidth(0.0)
    return lg

def top_legend(*args,fudge=0.1,**kwargs):
    '''
    Legend outside the plot on the top.

    Other Parameters
    ----------------
    fudge: padding between legend and axis, default -0.1
    '''
    lg = plt.legend(*args,**{**{
        'loc':'lower center',
        'bbox_to_anchor':(0.5,1.0+fudge),
        },**kwargs})
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
    data: np.array
    
    Returns
    -------
   : tuple
        ``np.min(data),np.max(data)``
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
    vmin=None,vmax=None,cm='afmhot',
    interpolation='nearest',dodraw=1):
    '''
    Plotting helper for continuous wavelet transform.
    
    Parameters
    ----------
    ff: numeric
        frequencies
    cwt: numeric
        wavelet transformed data (what orientation?)
    '''
    cwt  = np.squeeze(cwt)
    nf,N = np.shape(cwt)
    pwr  = np.abs(cwt)
    fest = ff[np.argmax(pwr,0)]
    plt.cla()
    imshow(pwr,aspect=aspect,extent=(0,N,ff[-1],ff[0]),
        vmin=vmin,vmax=vmax,
        interpolation=interpolation,cmap=cm)
    plt.xlim(0,N)
    plt.ylim(ff[0],ff[-1])
    try:
        plt.tight_layout()
    except:
        print('tight_layout missing, how old is your python? seriously')
    if dodraw:
        plt.draw()
        plt.show()

def plotWTPhase(ff,cwt,aspect=None,ip='nearest'):
    '''
    Plot the phase of a wavelet transform
    
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
    plt.xlim(0,N)
    plt.ylim(ff[0],ff[-1])
    tight_layout()
    draw()
    show()

wtpshow = plotWTPhase

def plotWTPhaseFig(ff,cwt,aspect=50,
    vmin=None,vmax=None,cm='bone',interpolation='nearest'):
    '''
    Plot the phase of a wavelet transform
    
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
    plt.tight_layout()
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


def good_colorbar(
    vmin=None,
    vmax=None,
    cmap=None,
    image=None,
    title='',
    ax=None,
    sideways=False,
    border=True,
    spacing=5,
    width=15,
    labelpad=10,
    fontsize=10,
    labelsize=None,
    scale=1.0,
    va='c',
    ha='c',
    **kwargs):
    '''
    Matplotlib's colorbar function is pretty bad. This is less bad.
    r'$\\mathrm{\\mu V}^2$'

    Parameters:
        vmin      (number) : min value for colormap
        vmax      (number) : mac value for colormap
        cmap      (colormap): what colormap to use
        title     (string) : Units for colormap
        ax        (axis)   : optional, defaults to plt.gca(). axis to which to add colorbar
        sideways  (bool)   : Flips the axis label sideways
        border    (bool)   : Draw border around colormap box? 
        spacing   (number) : distance from axis in pixels. defaults to 5
        width     (number) : width of colorbar in pixels. defaults to 15
        labelpad  (number) : padding between colorbar and title in pixels, defaults to 10
        fontsize  (number) : title font size, defaults to 10
        labelsize (number) : tick label font size, defaults to ``fontsize``
        scale     (float)  : height adjustment relative to parent axis, defaults to 1.0
        va        (str)    : vertical alignment; "bottom" ('b'), "center" ('c'), or "top" ('t')
        ha        (str)    : horizontal alignment; "left" ('l'), "center" ('c'), or "right" ('r')
    Returns:
        axis: colorbar axis
    '''
    if not image is None:
        if not (vmin is None and vmax is None and cmap is None):
            raise ValueError(
                'Passing an image object overrides the arguments '
                '(vmin,vmax,cmap); leave these empty if setting the '
                'image keyword.')
        vmin, vmax = image.get_clim()
        cmap = image.cmap
        
    if labelsize is None:
        labelsize = fontsize
    if type(vmin) == mpl.image.AxesImage:
        img  = vmin
        cmap = img.get_cmap()
        vmin = img.get_clim()[0]
        vmax = img.get_clim()[1]
        ax   = img.axes
    oldax = plt.gca() #remember previously active axis
    if ax is None: ax=plt.gca()
    
    # Determine units based on axis dimensions
    if sideways:
        SPACING = pixels_to_yfigureunits(spacing)
        CWIDTH  = pixels_to_yfigureunits(width)
    else:
        SPACING = pixels_to_xfigureunits(spacing)
        CWIDTH  = pixels_to_xfigureunits(width)    

    # Get axis bounding box information    
    bb = ax.get_position()
    x,y,w,h,r,l,t,b = bb.xmin,bb.ymin,bb.width,bb.height,bb.xmax,bb.xmin,bb.ymax,bb.ymin

    if sideways:
        # Alignment codes for horizontal colorbars
        x0 = {
            'l':lambda:l,
            'c':lambda:l+(w-w*scale)/2,
            'r':lambda:l+(w-w*scale)
        }[ha.lower()[0]]()
        cax = plt.axes((x0,b-SPACING,w*scale,CWIDTH),frameon=border)
        plt.sca(cax)
        plt.imshow(np.array([np.linspace(vmin,vmax,100)]),
            extent=(vmin,vmax,0,1),
            aspect='auto',
            origin='upper',
            cmap=cmap)
        noy()
        nicex()
        plt.text(
            np.mean(plt.xlim()),
            plt.ylim()[1]-pixels_to_yunits(labelpad,ax=cax),
            title,
            fontsize=fontsize,
            rotation=0,
            horizontalalignment='center',
            verticalalignment  ='top')
    else:
        # Alignment codes for vertical colorbars
        y0 = {
            #'b':lambda:b-h,
            #'c':lambda:b-(h+h*scale)/2,
            #'t':lambda:b-h*scale
            'b':lambda:b,
            'c':lambda:b+(h-h*scale)/2,
            't':lambda:b+(h-h*scale)
        }[va.lower()[0]]()
        cax = plt.axes((r+SPACING,y0,CWIDTH,h*scale),frameon=border)
        plt.sca(cax)
        plt.imshow(np.array([np.linspace(vmax,vmin,100)]).T,
            extent=(0,1,vmin,vmax),
            aspect='auto',
            origin='upper',
            cmap=cmap)
        nox()
        nicey()
        plt.text(
            plt.xlim()[1]+pixels_to_xunits(labelpad,ax=cax),
            np.mean(plt.ylim()),
            title,**(dict(
            fontsize=fontsize,
            rotation=90,
            horizontalalignment='left',
            multialignment='center',
            verticalalignment  ='center')|kwargs))
        cax.yaxis.set_label_position("right")
        cax.yaxis.tick_right()

    cax.tick_params('both', length=0, width=0, 
        labelsize=labelsize, which='major')

    plt.sca(oldax) #restore previously active axis
    return cax

def complex_axis(scale):
    '''
    Draws a nice complex-plane axis with LaTeX Re, Im labels.
    
    Parameters
    ----------
    scale: float
    '''
    plt.xlim(-scale,scale)
    plt.ylim(-scale,scale)
    nicexy()
    ybartext(0,r'$\Im(z)$','k','w',lw=1,color='k',outline=False)
    xbartext(0,r'$\Re(z)$','k','w',lw=1,color='k',outline=False,horizontalalignment='right')
    noaxis()
    xlabel(u'μV',fontname='DejaVu Sans',fontsize=12)
    ylabel(u'μV',fontname='DejaVu Sans',fontsize=12)
    xticks(xticks()[0],fontsize=12)
    yticks(yticks()[0],fontsize=12)
    force_aspect()

def subfigurelabel(x,fontsize=10,dx=39,dy=7,ax=None,bold=True,**kwargs):
    '''
    Parameters
    ----------
    x: label
    '''
    if ax is None: ax = plt.gca()
    fontproperties = {
        'fontsize':fontsize,
        'family':'Bitstream Vera Sans',
        'weight': 'bold' if bold else 'normal',
        'va':'bottom',
        'ha':'left'}
    fontproperties.update(kwargs)
    plt.text(
        plt.xlim()[0]-pixels_to_xunits(dx),
        plt.ylim()[1]+pixels_to_yunits(dy),
        x,
        **fontproperties)

def sigbar(x1,x2,y,
    pvalue=None,
    dy=5,
    padding=1,
    fontsize=10,
    color=BLACK,
    label_pvalue=True,
    **kwargs):
    '''
    Draw a significance bar between positions
    ``x1`` and ``x2`` at height ``y``. 
    
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
    label_pvalue: boolean; default True
        Label p-value in scientific notation
    **kwargs:
        Forwarded to the ``plot()`` command that draws the
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
        if label_pvalue:
            text(np.mean([x1,x2]),height+dy*padding,pvalue,
                fontsize=fontsize,
                horizontalalignment='center')

def hsigbar(y1,y2,x,
    pvalue=None,
    dx=5,
    padding=1,
    fontsize=10,
    color=BLACK,
    label_pvalue=True,
    **kwargs):
    '''
    Draw a significance bar between position y1 and y2 at 
    horizontal position x.
    
    Parameters
    ----------
    y1: float
    y2: float
    x : float
    
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
    label_pvalue: boolean; default True
        Label p-value in scientific notation
    **kwargs:
        Forwarded to the ``plot()`` command that draws the
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
        if label_pvalue:
            text(w+dx*padding,np.mean([y1,y2]),pvalue,
                fontsize=fontsize,ha='left',va='center')

def savefigure(name,stamp=True,**kwargs):
    '''
    Saves figure as both SVG and PDF, prepending the current date-ti,me
    in YYYYMMDD_HHMMSS format
    
    Parameters
    ----------
    name: string
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
    plt.savefig(dirname + os.path.sep+prefix+'.svg',
        transparent=True,bbox_inches='tight',pad_inches=0,**kwargs)
    plt.savefig(dirname + os.path.sep+prefix+'.pdf',
        transparent=True,bbox_inches='tight',pad_inches=0,**kwargs)
    plt.savefig(dirname + os.path.sep+prefix+'.png',
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
    y1,y2 = plt.ylim()
    precision = 10.0**precision
    _y1 = floor(y1*precision)/precision
    _y2 = ceil (y2*precision)/precision
    plt.ylim(min(_y1,plt.ylim()[0]),max(_y2,plt.ylim()[1]))

def round_to_precision(x,precision=1):
    '''
    Round to a specified number of significant figures
    
    Parameters
    ----------
    x: scalar
        Number to round
    precision: positive integer, default=1
        Number of digits to keep
    
    Returns
    -------
    x: scalar
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
    x: scalar
        Number to round
    precision: positive integer, default=1
        Number of digits to keep
    
    Returns
    -------
    x: scalar
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
    x: scalar
        Number to round
    precision: positive integer, default=1
        Number of digits to keep
    
    Returns
    -------
    x: scalar
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
    plt.ylim(min(_y1,plt.ylim()[0]),max(_y2,plt.ylim()[1]))


def Gaussian2D_covellipse(M,C,N=60,**kwargs):
    '''
    xy = Gaussian2D_covellipse(M,C,N=60,**kwargs)

    Plot a covariance ellipse for 2D Gaussian with mean M and covariance C
    Ellipse is drawn at 1 standard deviation

    Parameters
    ----------
    M: tuple of (x,y) coordinates for the mean
    C: 2x2 np.array-like covariance matrix
    N: optional, number of points in ellipse (default 60)

    Returns
    -------
    xy: list of points in the ellipse
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
    m: mean
    v: variance
    
    Other Parameters
    ----------------
    color: 
        Plot color
    alpha: 
        Shaded confidence alpha color blending value
    smooth: int
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

def yscalebar(
    ycenter,
    yheight,
    label,
    x=None,
    color='k',
    fontsize=9,
    ax=None,
    side='left',
    pad=2
    ):
    '''
    Add vertical scale bar to plot
    
    Parameters
    ----------
    ycenter
    yheight
    label
    
    Other Parameters
    ----------------
    x: number default None
    color: default 'k'
    fontsize: default 9
    ax: default None
    side: default 'left'
    pad: default 2
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
        plt.text(x-pixels_to_xunits(pad),np.mean(yspan),label,
            rotation=90,
            fontsize=fontsize,
            horizontalalignment='right',
            verticalalignment='center',
            clip_on=False)
    else:
        plt.text(x+pixels_to_xunits(pad),np.mean(yspan),label,
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
    90 : 4.605
    95 : 5.991
    97.5: 7.378
    99 : 9.210
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
    ax: axis, if None (default), uses the current axis.
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
    ax: axis, if None (default), uses the current axis.
    s: float; passed as the ``headlength`` and ``headwidth`` arrow property.
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
    Connect two points with a ``T`` (inhibition; braking) arrow.
    
    Parameters
    ----------
    x1: float
    y1: float
    x2: float
    y2: float

    Other Parameters
    ----------------
    ax: axis; if None (default), uses the current axis.
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
    
    Other Parameters
    ----------------
    color: matplotlib.color, default (.6,)*3
    '''
    # new clear axis overlay with 0-1 limits
    from matplotlib import pyplot, lines
    ax2 = pyplot.axes([0,0,1,1],facecolor=(1,1,1,0))
    x,y = np.array([[0,0,1,1,0], [0,1,1,0,0]])
    line = lines.Line2D(x, y, lw=1, color=color)
    ax2.add_line(line)
    plt.xticks([]); plt.yticks([]); noxyaxes()

def morexticks(ax=None):
    '''
    Add more ticks to the x axis

    Other Parameters
    ----------------
    ax: axis, if None (default), uses the current axis.
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

def moreyticks(ax=None):
    '''
    Add more ticks to the y axis

    Other Parameters
    ----------------
    ax: axis, if None (default), uses the current axis.
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

def moreticks(**kw):
    morexticks(**kw)
    moreyticks(**kw)

def border_width(lw=0.4,ax=None):
    '''
    Adjust width of axis border

    Parameters
    ----------
    lw: line width of axis borders to use

    Other Parameters
    ----------------
    ax: axis, if None (default), uses the current axis.
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
        bins  = np.array(sorted(x))[::skip]
        bins[-1] = np.max(y)+1e-9
    nbins = len(bins)-1
    means,stds,sems = [],[],[]
    Δe = (bins[1:]+bins[:-1])*0.5
    for i in range(nbins):
        ok = (x>=bins[i])&(x<bins[i+1])
        n  = np.sum(ok)+1
        v  = np.nanvar(y[ok])
        if not np.isfinite(v):
            v = 0
        m = np.nanmean(y[ok])
        if not np.isfinite(m):
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
    Plot a trend line
    
    Parameters
    ----------
    x: 
        x points
    y: 
        y points
    
    Other Parameters
    ----------------
    ax: 
        figure axis for plotting, if None uses plt.gca()
    '''
    if ax is None:
        ax = plt.gca()
    m,b = np.polyfit(x,y,1)
    xl = np.array(ax.get_xlim())
    plt.plot(xl,xl*m+b,
        label='offset = %0.2f\nslope = %0.2f'%(b,m),
        color=color)
    ax.set_xlim(*xl)
    plt.legend(edgecolor=(1,)*4)

def shellplot(x,y,z,SHELLS,
    label='',
    vmin=None,vmax=None,ax=None,doline=False,**opts):
    '''
    Averages X and Y based on bins of Z
    
    Parameters
    ----------
    x (1D np.ndarray): horizontal axis data
    y (1D np.ndarray): vertical axis data
    z (1D np.ndarray): time series to use to define groups
    SHELLS: number of groups to create
    
    Other Parameters
    ----------------
    label (str): Color bar title
    vmin (float): color bar minimum
    vmax (float): color bar maximum
    ax (axis): Matplotlib axis to plot into
    doline (bool): Draw linear trend line
    **opts (dict): Remaining keyword arguments forwarded to `plt.scatter`
    '''
    Xμ, σ, dμ, Δe = shellmean(z,x,bins=SHELLS)
    Yμ, σ, dμ, Δe = shellmean(z,y,bins=SHELLS)
    ok = np.isfinite(Xμ)
    Xμ = Xμ[ok]
    Yμ = Yμ[ok]
    Δe = Δe[ok]
    #if ax is None:
    #    smallplot()
    x = Xμ
    y = Yμ
    ns = len(x)
    plt.scatter(x,y,c=Δe,lw=0,s=16,vmin=vmin,vmax=vmax,**opts)
    cbar = plt.colorbar()
    simpleaxis()
    #cbar.set_ticks(arange(vmin,,2))
    cbar.ax.set_ylabel(label)
    if doline:
        trendline(x,y)
    return x,y,Δe

def arrow_between(A,B,size=None):
    '''
    Draw an arrow between two matplotlib axis instances
    
    Parameters
    ----------
    A: matplotlib.Axis
    B: matplotlib.Axis
    size: positive float; default None
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
    Split a 1D complex signal into real and imaginary 
    parts, setting components that are zero in either to 
    np.NaN. This lets us plot components separately without
    overlap (see ``plotz()``).
    
    Parameters
    ----------
    z: 
    thr: positive float; default 1e-9
    '''
    z   = np.complex64(z)
    r,i = np.float32(np.real(z)),np.float32(np.imag(z))
    r[abs(r)<thr]=np.NaN
    i[abs(i)<thr]=np.NaN
    return r,i

def plotz(x,z,thr=1e-9,**k):
    '''
    Plot a 1D complex signal, drawing the imaginary 
    component as a dashed line.
    
    Parameters
    ----------
    x:  
    z: 
    thr: positive float; default 1e-9
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
    ``__SAVE_LIMITS_PRIVATE_STORAGE__``.
    These can be restored later via ``restore_limits()``
    '''
    global __SAVE_LIMITS_PRIVATE_STORAGE__
    __SAVE_LIMITS_PRIVATE_STORAGE__ = (plt.xlim(),plt.ylim())


def restore_limits():
    '''
    Restore the ((x0,x1),(y0,y1) limits stored in 
    ``__SAVE_LIMITS_PRIVATE_STORAGE__``
    '''
    global __SAVE_LIMITS_PRIVATE_STORAGE__
    xl,yl = __SAVE_LIMITS_PRIVATE_STORAGE__
    plt.xlim(*xl)
    plt.ylim(*yl)
    

def mock_legend(names,colors=None,s=40,lw=0.6,marker='s',
    styles=None):
    '''
    For a list of (labels, colors), generate some 
    square scatter points outside the axis limits
    with the given labels, so that the ``legend()`` call
    will populate a list of labelled, colored squares.
    
    This does not actually draw the legend, but rather
    mock-up some off-screen labelled scatter points
    that can be used to populate the legend. 
    
    Use ``pyplot.legend()`` or one of the 
    ``neurotools.graphics.plot`` helpers
    ``nice_legend()``
    ``right_legend()``
    ``base_legend()``
    to draw the legend after calling this function.
    
    Parameters
    ----------
    labels: list of str
        List of labels to create
    colors: list of matplotlib.color
        List of label colors, same length as labels
        
        If a single color is given, I will use this for
        all markers. However, there is an edge case if 
        specifying a RGB tuple as a list with len(names)==3.
        This will cause an error. 
        
    Other Parameters
    ----------------
    s: int; default 40
        Size of markers. 
        Can also be a list of sizes.
    ls: float; default 0.6
        Line width for markers.
        Can also be a list of line width.
    marker: str; default 's' (square)
        Matplotlib marker character.
        Can also be a list of Matplotlib marker characters.
    styles: list of dictionaries
        To pass as keword arguments for each marker. 
        Overrides ALL other options. 
    '''
    names = [*names]
    save_limits()
    x0 = plt.xlim()[0]-100
    y0 = plt.ylim()[0]-100
    
    if not styles is None:
        for name,style in zip(names,styles):
            plt.scatter(x0,y0,label=name,**style)
        restore_limits()
        return
    
    if colors is None:
        colors = ['k',]*len(names)
    else:
        try:
            # Check if iterable
            colors = [*colors]
        except TypeError:
            colors = [colors,]*len(names)

    try:
        # Check if iterable
        marker = [*marker]
    except TypeError:
        marker = [marker,]*len(names)
    while len(marker)<len(names):
        marker += marker
    
    try:
        s = [*s]
    except TypeError:
        s = [s,]*len(names)
    while len(s)<len(names):
        s += s
    
    try:
        lw = [*lw]
    except TypeError:
        lw = [lw,]*len(names)
        
    for n,c,s,m,l in zip(names,colors,s,marker,lw):
        plt.scatter(x0,y0,s=s,color=c,marker=m,label=n,lw=l)
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
        If ``None``, uses ``pyplot.gca()``
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
        If ``None``, uses ``pyplot.gca()``
    which: str in {'major','minor','both'}; default 'both'
        Which set of ticks to apply to
    '''
    if ax is None:
        ax = plt.gca()
    ax.tick_params(axis='y', which=which, pad=pad)

def xticklen(l=0,w=None,ax=None,which='both',**kwargs):
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
        If ``None``, uses ``pyplot.gca()``
    which: str in {'major','minor','both'}; default 'both'
        Which set of ticks to apply to
    '''
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_tick_params(
        length=l, width=w, which=which, **kwargs)

def yticklen(l=0,w=None,ax=None,which='both',**kwargs):
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
        If ``None``, uses ``pyplot.gca()``
    which: str in {'major','minor','both'}; default 'both'
        Which set of ticks to apply to
    '''
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_tick_params(
        length=l, width=w, which=which, **kwargs)

def xin(ax=None,which='both'):
    '''
    Make x ticks point inward
    
    Other Parameters
    ----------------
    ax: matplotlib.axis or None; default None
        If ``None``, uses ``pyplot.gca()``
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
        If ``None``, uses ``pyplot.gca()``
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
        
    Other Parameters
    ----------------
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


def axvbands(widths,
    colors=BLACK,
    fade=0.8,
    startat=0,**
    kwargs):
    '''
    Wrapper for ``axvstripe`` that accepts band widths 
    rather than edges.
    
    Parameters
    ----------
    widths: list of numbers
        Width of each band
        
    Other Parameters
    ----------------
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


def zerohline(color='k',lw=None,**kwargs):
    '''
    Draw horizontal line at zero matching axis style.
    
    Other Parameters
    ----------------
    color: matplotlib.color; default 'k'
    lw: positive float
        Linewidth, if different from ``axes.linewidth``
    '''
    kwargs = dict(clip_on=False)|kwargs
    if lw is None:
        lw = matplotlib.rcParams['axes.linewidth']
    plt.axhline(0,color=color,lw=lw,**kwargs)


def zerovline(color='k',lw=None,**kwargs):
    '''
    Draw vertical line at zero matching axis style.
    
    Other Parameters
    ----------------
    color: matplotlib.color; default 'k'
    lw: positive float
        Linewidth, if different from ``axes.linewidth``
    '''
    if lw is None:
        lw = matplotlib.rcParams['axes.linewidth']
    plt.axvline(0,color=color,lw=lw,**kwargs)

def zerolines(**kw):
    '''
    Draw vertical and horizontal line at zero matching axis style.
    
    Other Parameters
    ----------------
    color: matplotlib.color; default 'k'
    lw: positive float
        Linewidth, if different from ``axes.linewidth``
    '''
    zerohline(**kw)
    zerovline(**kw)
    
    
def plot_circular_histogram(
    x,bins,r,scale,
    color=BLACK,alpha=0.8):
    '''
    Plot a circular histogram for data ``x``, 
    given in *degrees*. 
    
    Parameters
    ----------
    x: np.array
        Values to plot, in degrees
    bins: positive int
        Number of anugular bins to use
    r: positive float
        Radius of hisogram 
    scale: positive float
        Scale of histogram
    
    Other Parameters
    ----------------
    color: matplotlib.color; default BLACK
    alpha: float in [0,1]; default 0.8
    '''
    p,_ = histogram(x,bins,density=True)
    patches = []
    for j,pj in enumerate(p):
        base = r
        top  = r + pj*scale
        h1   = bins[j]
        h2   = bins[j]
        arc  = exp(1j*linspace(bins[j],bins[j+1],10)*pi/180)
        verts = []
        verts.extend(c2p(base*arc).T)
        verts.extend(c2p(top*arc[::-1]).T)
        patches.append(Polygon(verts,closed=True))
    collection = PatchCollection(patches,
        facecolors=color,
        edgecolors=WHITE,
        linewidths=0.5,
        alpha=alpha)
    gca().add_collection(collection)

def plot_quadrant(
    xlabel=r'←$\mathrm{noise}$→',
    ylabel=r'←$\varnothing$→'):
    '''
    Other Parameters
    ----------------
    xlabel: str; default '←$\\mathrm{noise}$→',
    ylabel: str; default '←$\\varnothing$→'
    '''
    # Quadrant of unit circle
    φ = linspace(0,pi/2,100)
    z = r*exp(1j*φ)
    x,y = c2p(z)
    plot(x,y,color='k',lw=1)
    plot([0,0],[0,r],color='k',lw=1)
    plot([0,r],[0,0],color='k',lw=1)
    text(r+.02,0,r'$0\degree$',ha='left',va='top')
    text(0,r+.02,r'$90\degree$',ha='right',va='bottom')
    text(0,r/2,ylabel,rotation=90,ha='right',va='center')
    text(r/2,0,xlabel,ha='center',va='top')

def quadrant_axes(q = 0.2):
    plt.xlim(0-q,1+q)
    plt.ylim(0-q,1+q)
    force_aspect()
    noaxis()
    noxyaxes()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

############################################################
# Boxplots

def nicebp(bp,color='k',linewidth=.5):
    '''
    Improve the appearance of a box and whiskers plot.
    To be called on the object returned by the matplotlib 
    boxplot function, with accompanying color information.
    
    Parameters
    ----------
    bp: point to boxplot object returned by matplotlib
    
    Other Parameters
    ----------------
    c: matplotlib.color; default ``'k'``
        Color to set boxes to
    linewidth: positive float; default 0.5
        Width of whisker lines. 
    '''
    for kk in 'boxes whiskers fliers caps'.split():
        setp(bp[kk], color=color)
    setp(bp['whiskers'],linestyle='solid',linewidth=linewidth)
    setp(bp['caps'],    linestyle='solid',linewidth=linewidth)


def colored_violin(data,position=1,color=RUST,edgecolor=WHITE,width=0.75,lw=0.8):
    vp = plt.violinplot(
        data,
        positions=[position],
        widths=[width],
        showextrema=True,
        showmedians=True,
    )
    for _ in vp['bodies']:
        _.set_facecolor(color)
        _.set_edgecolor(color)
        _.set_alpha(1)
        _.set_lw(lw/2)
    vp['cbars' ].set_color(color)
    vp['cmaxes'].set_color(color)
    vp['cmins' ].set_color(color)
    vp['cbars' ].set_lw(lw)
    vp['cmaxes'].set_lw(lw)
    vp['cmins' ].set_lw(lw)
    vp['cmedians'].set_color(edgecolor)
    vp['cmedians'].set_lw(lw)
    vp['cmedians'].set_capstyle("butt")

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
        Background color if ``filled=False``
    mediancolor: matplotlib.color; default None
        Defaults to BLACK unless color is BLACK, in which
        case it defaults to WHITE.
    **kwargs:
        Additional arguments fowarded to ``pyplot.boxplot()``
    '''
    if 'linewidth' in kwargs:
        lw = kwargs[linewidth]
    b = matplotlib.colors.to_hex(BLACK)
    if mediancolor is None:
        try:
            mediancolor = [
                BLACK if matplotlib.colors.to_hex(c)!=b \
                else WHITE for c in color]
        except:
            mediancolor = BLACK \
                if matplotlib.colors.to_hex(color)!=b \
                else WHITE
    kwargs2 = {k:v for (k,v) in kwargs.items() if not k.endswith('props')}
    bp = plt.boxplot(data,
        positions    = positions,
        patch_artist = True,
        showfliers   = showfliers,
        notch        = notch,
        whis         = whis, 
        medianprops  = {'lw':lw,'color':mediancolor}|kwargs.get('medianprops',{}),
        whiskerprops = {'lw':lw,'color':color}|kwargs.get('whiskerprops',{}),
        flierprops   = {'lw':lw,'color':color}|kwargs.get('flierprops',{}),
        capprops     = {'lw':lw,'color':color}|kwargs.get('capprops',{}),
        boxprops     = {'lw':lw,'color':color,
                  'facecolor':color if filled else bgcolor}|kwargs.get('boxprops',{}),
        **kwargs2);
    return bp

def boxplot_significance(
    a1,
    positionsa,
    b1=None,
    positionsb=None,
    fdr=0.05,
    dy=5,
    fontsize=6,
    label_pvalue=True,
    significance_mark='∗',
    paired=True):
    '''
    Perform Wilcoxon tests on a pair of box-plot sets
    and add significance brackets.
        
    Note that this, by default, assumed paired samples.
    Please set ``paired=False``. 
        
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
        Padding, in pixels, between box and p-value 
        annotation
    fontsize: postiive float; default 6
        Font size of p-value, if shown
    label_pvalue: boolean; default True
        Only for single-population tests.
        Whether to draw corrected p-value for significant 
        boxes.
    significance_mark: str; default '∗'
        Only for single-population tests.
        Marker to use to note significant boxes.    
    paired: boolean; default True
        Whether to compare conditions using paired or 
        unpaired tests.
        
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
    import neurotools.stats.pvalues as pvalues
    def test(*args):
        try: 
            if paired:
                return scipy.stats.wilcoxon(*args).pvalue
            else:
                return scipy.stats.mannwhitneyu(*args).pvalue
        except ValueError:
            return np.NaN
    
    if b1 is None:
        # One dataset: test if different from zero
        pv = np.float32([test(ai) for ai in a1]).ravel()
        pv2, reject = pvalues.correct_pvalues(
            pv,alpha=fdr,verbose=False)
        # Star heights
        hy = np.array([
                np.percentile(ai,95) for ai in a1
            ]).ravel()
        for ip,ir,x,y in zip(pv2,reject,positionsa,hy):
            if ip<fdr and label_pvalue:
                s = shortscientific(ip)+'\n'\
                  + significance_mark
                text(x,y+px2y(dy),s,
                    fontsize=fontsize,
                    ha='center')
    else:
        # Two datasets: test if different
        pv = np.float32([
                test(ai,bi) for ai,bi, in zip(a1,b1)
            ]).ravel()
        pv2, reject = pvalues.correct_pvalues(
            pv,alpha=fdr,verbose=False)
        # Bracket heights
        hy = np.array([[
            max(np.percentile(ai,95),np.percentile(bi,95)) 
            for ai,bi in zip(aa,bb)] 
            for aa,bb in [(a1,b1)]]).T.ravel()
        for ip,ir,x1,x2,y in zip(pv2,reject,positionsa,positionsb,hy):
            if ip<fdr:
                sigbar(x1,x2,y,
                    pvalue=ip,
                    dy=dy,
                    fontsize=fontsize,
                    label_pvalue=label_pvalue)
    return pv2, reject


def simplestem(x,y,**kwargs):
    '''
    Plot timeseries as verical lines dropped to y=0.
    Keyword arguments are forwarded to ``pyplot.plot()``.
    
    **Obsolete:** use ``matplotlib.pyplot.stem`` instead

    Parameters
    ----------
    x: 1D np.array
        X Location of bars
    y: 1D np.array
        Y position of bars
    '''
    x = np.array(x)
    y = np.array(y)
    y = np.array([y*0,y,np.NaN*y]).T.ravel()
    x = np.array([x,x,np.NaN*x]).T.ravel()
    plt.plot(x,y,**kwargs)
    

def pike(
    x,
    y,
    bins=10,
    bin_mode='percentile', # or 'uniform'
    point_function = 'mean', #'median'
    error_mode = 'pct', # 'std','pct','boot'
    error_range = [2.5,97.5],
    dolines=True,
    dopoints=True,
    doscatter=True,
    connect_points=False,
    doplot=True,
    color=None,
    linestyle={},
    pointstyle={},
    scatterstyle={},
    bin_offset=0.0,
    w = None,
    label=None,
    nbootstrap=500,
    sideways=False,
    ):
    '''
    Summarize data mean or median within bins. 
    
    Parameters
    ----------
    x: iterable
        x coordinates of data
    y: iterable
        y coordinates of data
    
    Other Parameters
    ----------------
    bins: positive int; default 10
    bin_mode: str; default 'percentile'
        ``'percentile'``: bins based on data percentiles'
        ``'uniform'``: evenly spaced bins between 
        ``min`` and ``max``.
    point_function: str; default 'mean'
        ``'mean'`` or ``'median'``
    error_mode: str; default 'p'
        ``'e'`` Standard error of mean;
        ``'s'`` Standard deviation of data;
        ``'p'`` Percentiles of data;
        ``'b'`` Bootstrap error of mean or median.
    error_range: tuple of numbers ∈(0,100); default (2.5,97.5)
        Lower and upper percentiles to use for error bars.
    dolines:boolean; default True
        Whether to draw error lines.
    dopoints:boolean; default True
        Whether to draw mean/median points.
    doscatter:boolean; default True
        Wheter to plot ``(x,y)`` as scatter points.
    connect_points: boolean; default False
        Whether to connect the points in a line plot.
    doplot:boolean; default True
        Whether to draw a plot
    linestyle: dict
        Keyword arguments for line style
    pointstyle: dict
        Keyword arguments for mean/median style
    scatterstyle: dict
        Keyword arguments for scatter points style
    bin_offset: float
        Fraction of median inter-bin distance to offset
        the error lins. This adjustment allows plotting of
        multiple series atop each-other without overlap.
    nbootstrap: int; default 200
        Number of bootstrap samples
    sideways: boolean, default False
        Flips the role of the `x` and `y` axes
    
    Returns
    -------
    xc: list
        ``x`` bin centers
    points: list
        mean or median of ``y`` in each bin
    lo: list
        lower data or error percentile in each bin
    hi: list
        upper data or error percentile in each bin
    '''
    import neurotools.stats.pvalues as pvalues

    # Data cleaning: get error range
    error_range = np.float32(error_range).ravel()
    assert np.all(error_range)>=0
    assert np.all(error_range)<=100
    plo,phi = sorted(error_range)[:2]
    
    # unravel and remove NaN
    y = np.array(y).ravel()
    x = np.array(x).ravel()
    ok = (np.isfinite(x)) & (np.isfinite(y))
    x = x[ok]
    y = y[ok]
    
    if not color is None:
        linestyle    = { 'color':color,**linestyle    }
        pointstyle   = { 'color':color,**pointstyle   }
        scatterstyle = { 'color':color,**scatterstyle }
    
    # Compute bins
    eps = 1e-9
    bin_mode = str(bin_mode).lower()[0]
    
    if isinstance(bins,int):
        if bin_mode=='p':#'percentile':
            # Adjust bins to contain similar numbers of points
            bins = np.nanpercentile(x,np.linspace(0,100,bins+1))
            bins[-1] += eps
        elif bin_mode=='u':#'uniform':
            # Assign to uniformly spaced bins
            bins = np.linspace(np.nanmin(x)-eps,np.nanmax(x)+eps,bins+1)
        else:
            raise ValueError(
                "``bin_mode`` must be 'percentile' or 'uniform'")
    nbins = len(bins)-1
    xc = centers(bins)
    
    # Adjustment allows plotting multiple series without
    # overlap
    xc += np.median(np.diff(xc))*bin_offset
    
    # Group data by bins 
    j = np.digitize(x,bins)-1
    groups = [y[i==j] for i in range(nbins)]
    
    '''
    if w is None:
        w = np.ones(x.shape)
    else:
        w = np.array(w).ravel()[ok]
    wg     = [w[i==j] for i in range(nbins)]
    '''
    str(point_function).lower()
    if point_function=='mean':
        mfun = np.nanmean
    elif point_function=='median':
        mfun = np.nanmedian
    else:
        #mfun = point_function
        raise ValueError(
            "``point_function`` must be 'mean' or 'median'")
        
    # Calculate measure of central tendency
    points = np.float32([*map(mfun,groups)])
    
    # Calculate error bars 
    error_mode = str(error_mode).lower()[0]
    if error_mode=='e':# standard error
        # Gaussian confidence intervals on mean
        means  = np.float32([*map(np.nanmean,groups)])
        counts = np.float32([np.sum(~np.isnan(g)) for g in groups])
        stds   = np.float32([*map(np.nanstd ,groups)])
        sems   = stds/np.sqrt(counts)
        lo = means + sems * scipy.stats.norm.ppf(plo/100)
        hi = means + sems * scipy.stats.norm.ppf(phi/100)
    elif error_mode=='s':
        # Gaussian model of data variance
        means  = np.float32([*map(np.nanmean,groups)])
        stds   = np.float32([*map(np.nanstd ,groups)])
        lo = means + stds * scipy.stats.norm.ppf(plo/100)
        hi = means + stds * scipy.stats.norm.ppf(phi/100)
    elif error_mode=='p':
        # Percentile model of data variace
        bounds = [
            np.nanpercentile(yi,[plo,phi])
            for yi in groups]
        bounds = [
            b if np.all(np.isfinite(b)) else (np.NaN,np.NaN)
            for b in bounds]
        lo,hi = zip(*bounds)
    elif error_mode=='b':
        # Bootstrap mean or median
        lo,hi = zip(*[
            np.nanpercentile(
                pvalues.bootstrap_statistic(
                    mfun,
                    yi,
                    nbootstrap),
                [plo,phi]
            ) 
            for yi in groups])
    else:
        raise ValueError(
            "``error_mode`` must be 'e', 's', 'p', "
            " or 'b'")
        
    lo = np.float32(lo)
    hi = np.float32(hi)
    if doplot:
        if doscatter:
            if sideways:
                plt.scatter(y,x,**{**{
                    'marker':'.',
                    'lw':0,
                    'color':AZURE,
                    's':20
                },**scatterstyle})
            else:
                plt.scatter(x,y,**{**{
                    'marker':'.',
                    'lw':0,
                    'color':AZURE,
                    's':20
                },**scatterstyle})
        if dolines:
            px = np.array([xc,xc,np.NaN*xc]).T.ravel()
            py = np.array([lo,hi,np.NaN*lo]).T.ravel()
            if sideways:
                plt.plot(py,px,**{**{
                    'lw':1,
                    'color':BLACK,
                    'solid_capstyle':'butt'
                },**linestyle})
            else:
                plt.plot(px,py,**{**{
                    'lw':1,
                    'color':BLACK,
                    'solid_capstyle':'butt'
                },**linestyle})
        if dopoints:
            if sideways:
                plt.scatter(points,xc,**{**{
                    'label':label,
                    'marker':'s',
                    'lw':0,
                    'color':BLACK,
                    's':10
                },**pointstyle})
            else:
                plt.scatter(xc,points,**{**{
                    'label':label,
                    'marker':'s',
                    'lw':0,
                    'color':BLACK,
                    's':10
                },**pointstyle})
        if connect_points: 
            if sideways:
                plt.plot(points,xc,**{**{
                    'lw':1.5,
                    'color':BLACK,
                    'solid_capstyle':'butt'
                },**linestyle})
            else:
                plt.plot(xc,points,**{**{
                    'lw':1.5,
                    'color':BLACK,
                    'solid_capstyle':'butt'
                },**linestyle})
            
    return xc,points,lo,hi

def confidencebox(x,y,
    median=None,boxcolor=RUST,w=.5,**kwargs):
    '''
    Use a box plot to draw a 2.5–97.5% confidence interval, 
    taking care of some tegious arguments.
    
    Parameters
    ----------
    x: number
        x location of box
    y: iterable
        samples from which to draw the box
    median: number
        central value whose confidence interval this
        box plot represents
        
    Other Parameters
    ----------------
    boxcolor: matplotlib.color; default RUST
        Box's color
    **kwargs: dictionary
        Overrides for any boxplot arguments. 
        See ``pyplot.boxplot()`` for more details. 
    '''
    props={
        'sym':'*',
        'vert':True,
        'whis':(2.5,97.5),
        'positions':[x],
        'widths':[w],
        'patch_artist':True,
        'usermedians':None if median is None else [median],
        'showmeans':False,
        'showcaps':True,
        'showbox':True,
        'showfliers':False,
        'boxprops':{'color':boxcolor,'facecolor':boxcolor},
        'capprops':{'color':boxcolor},
        'whiskerprops':{'color':boxcolor},
        'flierprops':{'color':boxcolor},
        'medianprops':{'color':'k'}
    }
    props = {**props,**kwargs}
    bp = plt.boxplot(y,**props)
    for cap in bp['caps']:
        if props['vert']:
            x = np.mean(cap.get_xdata())
            cap.set_xdata(x + np.array([-w/2,w/2]))
        else:
            y = np.mean(cap.get_ydata())
            cap.set_ydata(y + np.array([-w/2,w/2]))

    
def anatomy_axis(
    x0,
    y0,
    dx=15,
    dy=None,
    tx=4,
    ty=None,
    l='M',
    r='L',
    u='A',
    d='P',
    fontsize=8
    ):
    '''
    Draw an anatomical axis crosshairs on the current axis.
    
    Parameters
    ----------
    x0: number
        X coordinate of anatomy axis.
    y0: number
        Y coordinate of anatomy axis.
    dx: positive number; default 15
        Width of axis in pixels
    dy: positive number; default dx
        Height of axis in pixels
    tx: positive number; default 4
        Horizontal label padding in pixels
    ty: positive number; default tx
        Vertical label padding in pixels
    l: str; default 'M'
        Label for leftwards direction
    r: str; default 'L'
        Label for rightwards direction
    u: str; default 'A'
        Label for upwards direction
    d: str; default 'P'
        Label for downwards direction
    fontsize: positive number; 8
        Font size for labels
    '''
    if dy is None: dy=dx
    if ty is None: ty=tx
    dx = px2x(dx)
    dy = px2y(dy)
    tx = px2x(tx)
    ty = px2y(ty)
    plt.plot([x0-dx,x0+dx],[y0,y0],
        color='k',lw=0.6,clip_on=False)
    plt.plot([x0,x0],[y0-dy,y0+dy],
        color='k',lw=0.6,clip_on=False)
    plt.text(x0+dx+tx,y0,r,
        fontsize=fontsize,
        ha='left',va='center',clip_on=False)
    plt.text(x0-dx-tx,y0,l,
        fontsize=fontsize,
        ha='right',va='center',clip_on=False)
    plt.text(x0,y0+dy+ty,u,
        fontsize=fontsize,
        ha='center',va='bottom',clip_on=False)
    plt.text(x0,y0-dy-ty,d,
        fontsize=fontsize,
        ha='center',va='top',clip_on=False)


def vpaired_histogram(
    ax1,
    ax2,
    x1,
    x2,
    bins = 10,
    color1='teal',
    color2='orange',
    name1='',
    name2='',
    label = '← (density) →',
    labelpad = 20,
    labelproperties = {},
    ):
    '''
    Draw a veritically paired histogram.
    
    >>> vpaired_histogram(subplot(211), subplot(212), randn(100), randn(100))
    
    Parameters
    ----------
    ax1: axis
        The top axis
    ax2: axis
        The bottom axis. This will be shifted to 
        abut the top axis and match its width.
    x1: iterable of numbers
        Data for top histogram
    x2: iterable of numbers
        Data for bottom histogram
    
    Other Parameters
    ----------------
    bins: int or series; default 10
        Histogram bin edges or number of histogram bins.
    color1: matplotlib color
        Color of top histogram
    color2: matplotlib color
        Color of bottom histogram
    name1: str
        Label for top histogram
    name2: str
        Label for bottom histogram
    label: str
        Y axis label
    labelpad: int
        Y axis label spacing in pixels
    labelproperties: dict
        Keyword arguments forwarded to figure.text()
        for drawing the Y axis label.
    '''
    
    x1 = [*x1]
    x2 = [*x2]
    xlo = min(np.min(x1),np.min(x2))
    xhi = max(np.max(x1),np.max(x2))
    if isinstance(bins,int):
        bins = np.linspace(xlo,xhi,bins+1)
    
    vtwin(ax1,ax2)
    y1 = -inf
    for x,ax,color,name in zip((x1,x2),(ax1,ax2),(color1,color2),(name1,name2)):
        plt.sca(ax)    
        plt.hist(x,bins,color=color,label=name,density=True)
        y1 = max(y1,plt.ylim()[1])
    
    # Same x range
    ax1.set_xlim(xlo,xhi)
    ax2.set_xlim(xlo,xhi)
    
    # Remove excess axis spines
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax2.get_yaxis().tick_left()
    
    # Avoid duplicate zero on y axis
    ax1.set_ylim(0,y1)
    ax2.set_ylim(y1,0)
    yt = ax1.get_yticks()
    ax1.set_yticks([i for i in yt if not i==0.0])
    ax2.set_yticks(yt)
    
    # It's hard to draw a centered label across axes
    # Take care of this. 
    dx = pixels_to_xfigureunits(labelpad)
    x = ax1.get_position().xmin - dx
    y = (ax1.get_position().ymin + ax2.get_position().ymax)/2
    plt.gcf().text(x-0.01,y,label,**{
        **dict(
            fontsize=11,
            ha='right',
            va='center',
            rotation=90),
        **labelproperties})  
        














from matplotlib.patches    import PathPatch
from matplotlib.path       import Path
from scipy.stats           import vonmises
from numpy                 import linspace, exp

def hollow_polygon(outer,inner,**kwargs):
    '''
    Add a polygon with a single interior region 
    subtracted to the current plot.
    '''
    gca().add_patch(PathPatch(Path(
        [*outer] + [*inner][::-1],
        [Path.MOVETO] + [Path.LINETO]*(len(outer)-2) + 
        [Path.CLOSEPOLY] + 
        [Path.MOVETO] + 
        [Path.LINETO]*(len(inner)-2) + [Path.CLOSEPOLY]
    ),**kwargs))

def add_polygon(points,**kwargs):
    '''
    Add a closed polygon to the current plot.
    '''
    gca().add_patch(PathPatch(Path(
        [*points],
        [Path.MOVETO] + 
        [Path.LINETO]*(len(points)-2) + 
        [Path.CLOSEPOLY]
    ),**kwargs))

def linscale_polar_plot(
    h,
    r,
    rinner,
    router,
    lw=0.8,
    facecolor=lighten(BLACK,.75),
    edgecolor=lighten(BLACK,.5),
    alpha=1.0,
    ntheta=360,
    zorder=-500,
    flipud=False,
    clip_on=False,
    ):
    from neurotools.signal import unitscale 
    from neurotools.util.tools import c2p
    z = exp(-1j*h) if flipud else exp(1j*h)
    inner = c2p(z*rinner)
    outer = c2p(z*(rinner+(router-rinner)*unitscale(r)))
    # Shaded region
    hollow_polygon(
        outer.T,
        inner.T,
        facecolor=facecolor,
        lw=0,
        zorder=zorder,
        alpha=alpha,
        clip_on=clip_on
    )
    # Outer line
    if lw>0:
        plot(
            *outer,
            color=edgecolor,
            lw=lw,
            zorder=zorder,
            clip_on=clip_on
        )


def vonmises_ring(
    kappa,
    loc,
    rinner,
    router,
    lw=0.8,
    facecolor=lighten(BLACK,.75),
    edgecolor=lighten(BLACK,.5),
    color=None,
    alpha=1.0,
    ntheta=360,
    draw_mean=True,
    markerprops={},
    zorder=-500,
    flipud=False,
    clip_on=False,
    ):
    '''
    Plot a von Mises distribution in polar coordinates,
    with inverse-dispersion parameter ``kappa``,
    location parameter ``loc`` (mean angle in radians),
    baseline at ``rinner``, and peak at ``router``.
    '''
    from neurotools.signal import unitscale 
    from neurotools.util.tools import c2p
    
    if not color is None:
        facecolor = edgecolor = color
    
    h = linspace(0,2*np.pi,ntheta+1)
    z = exp(-1j*h) if flipud else exp(1j*h)
    
    # von Mises distribution shaded ring
    vm = scipy.stats.vonmises(kappa, loc, 1)
    rr = unitscale(vm.pdf(h))
    vminner = c2p(z*rinner)
    vmouter = c2p(z*(rinner+(router-rinner)*unitscale(rr)))
    
    # Shaded region
    hollow_polygon(
        vmouter.T,
        vminner.T,
        facecolor=facecolor,
        lw=0,
        alpha=alpha,
        zorder=zorder,
        clip_on=clip_on
    )
    # Outer line
    if lw>0:
        plot(
            *vmouter,
            color=edgecolor,
            lw=lw,
            zorder=zorder,
            clip_on=clip_on
        )
    # dial tick for mean
    if draw_mean:
        z = exp(1j*loc)
        plot(
            *c2p([z*rinner,z*router]),
            **(dict(
                zorder=zorder,
                clip_on=clip_on,
                lw=2.0,
                color=[0.26666668, 0.32156864, 0.36078432],
                solid_capstyle='butt',
            )|markerprops)
        )

    
def disk_axis(
    r1,
    r2,
    nspokes    = 12,
    facecolor  = OFFWHITE,
    edgecolor  = OFFBLACK,
    lw         = 0.4,
    zorder     = -100,
    fix_limits = True,
    clip_on    = False,
    draw_inner = True,
    draw_outer = True,
    ):
    '''
    Disk axis: Draw a disk with alternating white/colored
    sectors like a roulette wheel. 
    
    Parameters
    ----------
    r1: positive float
        Radius of lower axis limit, ``r1>0``.
    r2: positive float
        Radius of upper axis limit, ``r2>r1``.
        
    Other Parameters
    ----------------
    nspokes: non-negative integer
        Number of bands/spokes to draw, default is 12.
    facecolor: matplotlib color
        Color of bands/spokes; Default is `OFFWHITE` defined in
        `neurotools.graphics.color`.
    edgecolor: matplotlib color
        Color of the upper/lower radius axes; 
        Default is `OFFBLACK` defined in `neurotools.graphics.color`.
    lw: positive float, default 0.4
    zorder: numeric, default -100
    fix_limits: boolean, default True
    clip_on: boolean, default False
    draw_inner: boolean, default True
    draw_outer: boolean, default True
    '''
    from neurotools.util.tools import c2p
    
    # Spokes
    for i in range(nspokes):
        z = exp(1j*linspace(
            i/nspokes*2*pi,
            (i+.5)/nspokes*2*pi,
            100))
        outer = c2p(z*(r2)).T
        inner = c2p(z*(r1)).T
        add_polygon(
            [*outer] + [*inner][::-1],
            facecolor = facecolor,
            edgecolor = (1,1,1,0),
            lw        = 0.0,
            zorder    = zorder,
            clip_on   = clip_on
        )
    
    # ring
    if lw>0:
        z = exp(1j*linspace(0,2*np.pi,361))
        if draw_inner:
            plot(
                *c2p(z*r1),
                lw=lw,
                color=edgecolor,
                clip_on=clip_on,
                zorder=zorder+1
                )
        if draw_outer:
            plot(
                *c2p(z*r2),
                lw=lw,
                color=edgecolor,
                clip_on=clip_on,
                zorder=zorder+1
                )
            
    if fix_limits:
        #x0,x1 = plt.xlim()
        #y0,y1 = plt.ylim()
        #x0 = min(x0,-r2)
        #y0 = min(y0,-r2)
        #x1 = max(x1, r2)
        #y1 = max(y1, r2)
        #plt.xlim(x0,x1)
        #plt.ylim(y0,y1)
        plt.xlim(-r2,r2)
        plt.ylim(-r2,r2)
        force_aspect()
    
    
    
    
