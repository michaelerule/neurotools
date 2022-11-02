#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Miscellaneous color-related functions.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function


import math
import pylab

from neurotools.signal import gaussian_smooth
from neurotools.signal import circular_gaussian_smooth
from neurotools import signal

import numpy as np
from   numpy import pi

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from os.path    import expanduser
from matplotlib import cm

# Some custom colors, just for fun! 
WHITE      = np.float32(mpl.colors.to_rgb('#f1f0e9'))
RUST       = np.float32(mpl.colors.to_rgb('#eb7a59'))
OCHRE      = np.float32(mpl.colors.to_rgb('#eea300'))
AZURE      = np.float32(mpl.colors.to_rgb('#5aa0df'))
TURQUOISE  = np.float32(mpl.colors.to_rgb('#00bac9'))
TEAL       = np.float32(mpl.colors.to_rgb('#00bac9'))
BLACK      = np.float32(mpl.colors.to_rgb('#44525c'))
YELLOW     = np.float32(mpl.colors.to_rgb('#efcd2b'))
INDIGO     = np.float32(mpl.colors.to_rgb('#606ec3'))
VIOLET     = np.float32(mpl.colors.to_rgb('#8d5ccd'))
MAUVE      = np.float32(mpl.colors.to_rgb('#b56ab6'))
MAGENTA    = np.float32(mpl.colors.to_rgb('#cc79a7'))
CHARTREUSE = np.float32(mpl.colors.to_rgb('#b59f1a'))
MOSS       = np.float32(mpl.colors.to_rgb('#77ae64'))
VIRIDIAN   = np.float32(mpl.colors.to_rgb('#11be8d'))
CRIMSON    = np.float32(mpl.colors.to_rgb('#b41d4d'))
GOLD       = np.float32(mpl.colors.to_rgb('#ffd92e'))
TAN        = np.float32(mpl.colors.to_rgb('#765931'))
SALMON     = np.float32(mpl.colors.to_rgb('#fa8c61'))
GRAY       = np.float32(mpl.colors.to_rgb('#b3b3b3'))
LICHEN     = np.float32(mpl.colors.to_rgb('#63c2a3'))
RUTTEN  = [GOLD,TAN,SALMON,GRAY,LICHEN]
GATHER  = [WHITE,RUST,OCHRE,AZURE,TURQUOISE,BLACK]
COLORS  = [BLACK,WHITE,YELLOW,OCHRE,CHARTREUSE,MOSS,VIRIDIAN,TURQUOISE,AZURE,INDIGO,VIOLET,MAUVE,MAGENTA,RUST]
CYCLE   = [BLACK,RUST,TURQUOISE,OCHRE,AZURE,MAUVE,YELLOW,INDIGO]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CYCLE)

######################################################################
# Hex codes


def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)


def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))

######################################################################
# Hue / Saturation / Luminance color space code


def hsv2rgb(h,s,v,force_luminance=None,method='perceived'):
    '''
    Convert HSV colors to RGB 
    
    Parameters
    ----------
    h: hue 0 to 360
    s: saturation 0 to 1
    v: value 0 to 1
    
    Returns
    -------
    r,g,b: RGB tuple
    '''
    h,s,v = map(float,(h,s,v))
    h60 = h/60.0
    h60f = math.floor(h60)
    hi = int(h60f)%6
    f = h60-h60f
    p = v*(1-s)
    q = v*(1-f*s)
    t = v*(1-(1-f)*s)
    v = min(1.,max(0.,v))
    t = min(1.,max(0.,t))
    p = min(1.,max(0.,p))
    q = min(1.,max(0.,q))
    RGB = ((v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q))[hi]
    RGB = np.array(RGB)
    if not force_luminance is None:
        Lmtx = luminance_matrix(method)
        for i in range(10):
            lum    = np.dot(Lmtx,RGB)
            delta  = (force_luminance-lum)*0.1
            newRGB = np.clip(RGB+delta,0,1)
            newRGB = np.clip(newRGB*(force_luminance/np.dot(Lmtx,newRGB)),0,1)
            if np.allclose(RGB,newRGB):
                break
            RGB = newRGB
            
    return RGB

def lightness(r,g,b,method='lightness'):
    '''
    Parameters
    ----------
    r: red
    g: green
    b: blue
    method: str
        Can be 'perceived', 'standard', or 'lightness'. 'lightness' is the 
        default

    Returns
    -------
    M: np.ndarray
        Weights to convert RGB vectors into a luminance value.
    '''
    return luminance_matrix(method=method).dot((r,g,b))


def luminance_matrix(method='perceived'):
    '''
    Method 'perceived': .299*R + .587*G + .114*B
    Method 'standard' : .2126*R + .7152*G + .0722*B
    Method 'lightness': .3*R + .59*G + .11*B
    
    Parameters
    ----------
    method: str
        Can be 'perceived', 'standard', or 'lightness'. 
        'perceived' is the default.
    
    Returns
    -------
    LRGB : np.array
        Weights for converting RGB vectors to scalar luminance values.
    '''
    methods = {
        'standard' :[.2126,.7152,.0722],
        'perceived':[.299,.587,.114],
        'lightness':[.30,.59,.11]
        }
    if not method in methods:
        raise ValueError('Method should be standard, perceived, or lightness')
    
    LRGB = np.array(methods[method])
    LRGB = LRGB / np.sum(LRGB)
    return LRGB


def match_luminance(target,color,
    THRESHOLD=0.01,squared=False,method='perceived'):
    '''
    Adjust color to match luminosity of target

    Method 'perceived' .299*R + .587*G + .114*B
    Method 'standard'  .2126*R + .7152*G + .0722*B
    Method 'lightness' .3*R + .59*G + .11*B
    
    Parameters
    ----------
    target:
    color:
    
    Other Parameters
    ----------------
    THRESHOLD: 0.01
    squared: False
    method: 'perceived'
    
    Returns
    -------
    color
    '''
    LRGB   = luminance_matrix(method)
    color  = np.array(color)
    target = np.array(target)
    if squared:
        luminance = np.dot(LRGB,target**2)**0.5
        source    = np.dot(LRGB,color**2)**0.5
    else:
        luminance = np.dot(LRGB,target)
        source    = np.dot(LRGB,color)

    luminance = np.sum(luminance)
    # do a bounded relaxation of the color to attempt to tweak luminance
    # while preserving hue, saturation as much as is possible
    while abs(source-luminance)>THRESHOLD:
        arithmetic_corrected = np.clip(color+luminance-source,0,1)
        geometric_corrected  = np.clip(color*luminance/source,0,1)
        correction = .5*(arithmetic_corrected+geometric_corrected)
        color = .9*color+0.1*np.clip(correction,0,1)
        if squared:
            source = np.dot(LRGB,color**2)**0.5
        else:
            source = np.dot(LRGB,color)
    return color


def rotate(colors,th):
    '''
    Rotate a list of rgb colors by angle theta
    
    Parameters
    ----------
    colors : array-like
        Iterable of (r,g,b) tuples. Rotation does not affect magnitude, 
        but may cause resulting colors to fall outside original colorspace.
    th : float
        Angle to rotate
    
    Returns
    -------
    list : results
        List of hue-rotated (r,g,b) tuples
    '''
    Q1 = np.sin(th)/np.sqrt(3)
    Q2 = (1-np.cos(th))/3
    results = []
    for (r,g,b) in colors:
        rb = r-b;
        gr = g-r;
        bg = b-g;
        r1 = Q2*(gr-rb)-Q1*bg+r;
        Z  = Q2*(bg-rb)+Q1*gr;
        g += Z + (r-r1);
        b -= Z;
        r = r1;
        results.append((r,g,b))
    return results


def RGBtoHCL(r,g,b,method='perceived'):
    '''
    Convert RGB colors to Hue, Chroma, Luminance color space

    Parameters
    ----------
    r:
    g:
    b:

    Returns
    -------
    hue:
    chroma:
    luminance:
    '''
    alpha  = .5*(2*r-g-b)
    beta   = np.sqrt(3)/2*(g-b)
    hue    = np.arctan2(beta,alpha)
    chroma = np.sqrt(alpha**2+beta**2)
    L = lightness(r,g,b)
    return hue,chroma,L


def hue_angle(c1,c2):
    '''
    Calculates the angular difference in hue between two colors in Hue, 
    Chroma, Luminance (HCL) colorspace.
    
    Parameters
    ----------
    c1: first color
    c2: second color
    
    Returns
    -------
    hue angle diference
    '''
    H1 = RGBtoHCL(*c1)[0]
    H2 = RGBtoHCL(*c2)[0]
    return H2-H1


def hcl2rgb(h,c,l,target=None, method='perceived'):
    '''
    
    Parameters
    ----------
    h: hue in degrees
    c: chroma
    l: luminosity
    
    Returns
    -------
    RGB: ndarray
        Length-3 numpy array of RGB color components
    '''
    LRGB     = luminance_matrix(method)
    x1,x2,x3 = LRGB
    h     = h*pi/180.0
    alpha = np.cos(h)
    beta  = np.sin(h)*2/np.sqrt(3)
    B     = l - x1*(alpha+beta/2)-x2*beta
    R     = alpha + beta/2 + B
    G     = beta + B
    RGB   = np.array([R,G,B])
    RGB   = np.clip(RGB,0,1)
    #if not target==None:
    # Try to correct for effect of clipping
    #RGB = match_luminance(target,RGB,method=method)
    if not target is None:
        Lmtx = luminance_matrix(method)
        reference = np.ones(3,'f')
        reference = reference*(target/np.dot(Lmtx,reference))
        for i in range(5):
            lum    = np.dot(Lmtx,RGB)
            delta  = (target-lum)*0.1
            newRGB = np.array(RGB)
            if any( abs(abs(RGB-0.5*2)-1.0) < 0.001):
                newRGB = newRGB*.95+0.05*reference
            newRGB = np.clip(newRGB*(target/np.dot(Lmtx,newRGB)),0,1)
            if np.allclose(RGB,newRGB):
                break
            RGB = newRGB
    
    return RGB


def circularly_smooth_colormap(cm,s):
    '''
    Smooth a colormap with cirular boundary conditions

    s: sigma, standard dev of gaussian smoothing kernel in samples
    cm: color map, array-like of RGB tuples
    
    Parameters
    ----------
    cm : colormap
    s : smoothing radius

    Returns
    -------
    RBG : np.ndarray
        Colormap smoothed with circular boundary conditions.
    '''
    # Do circular boundary conditions the lazy way
    cm = np.array(cm)
    N = cm.shape[0]
    R,G,B = cm.T
    R = circular_gaussian_smooth(R,s)
    G = circular_gaussian_smooth(G,s)
    B = circular_gaussian_smooth(B,s)
    RGB = np.array([R,G,B]).T
    #return np.array([np.fft.fftshift(c) for c in RGB.T]).T
    return RGB


def isoluminance1(h,l=.5):
    '''
    
    Parameters
    ----------
    h : hue, float 
    
    Returns
    -------
    '''
    return hcl2rgb(h,1,l,target=float(l))


def isoluminance2(h):
    '''
    
    Parameters
    ----------
    h : hue, float 
    
    Returns
    -------
    '''
    return hcl2rgb(h,1,1.0,target=.5)*(1+(h%5))/5


def isoluminance3(h):
    '''
    
    Parameters
    ----------
    h : hue, float 
    
    Returns
    -------
    '''
    return hcl2rgb(h,1,1.0,target=.5)*(1+(h%15))/15


def isoluminance4(h):
    '''
    
    Parameters
    ----------
    h : hue, float 
    
    Returns
    -------
    '''
    return hcl2rgb(h,1,1.0,target=.5)*(1+(h%60))/60


def lighthues(N=10,l=0.7):
    '''
    
    Parameters
    ----------
    N : 
    l : luminance in [0,1]
    
    Returns
    -------
    '''
    return [isoluminance1(h,l) for h in np.linspace(0,360,N+1)[:-1]]


def darkhues(N=10,l=0.4):
    '''
    
    Parameters
    ----------
    N:
    l:
    
    Returns
    -------
    np.ndarray
    '''
    return [isoluminance1(h,l) for h in np.linspace(0,360,N+1)[:-1]]


def medhues(N=10,l=0.6):
    '''
    
    Parameters
    ----------
    N
    l
    
    Returns
    -------
    '''
    return [isoluminance1(h,l) for h in np.linspace(0,360,N+1)[:-1]]


def radl2rgb(h,l=1.0):
    '''
    Slightly more optimized HSL conversion routine.
    Saturation fixed at 1
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    x1 = .30
    x2 = .59
    x3 = .11
    LRGB  = np.array([x1,x2,x3])
    alpha = np.cos(h)
    beta  = np.sin(h)*2/sqrt(3)
    B = 1.0 - x1*(alpha+beta/2)-x2*beta
    R = alpha + beta/2+B
    G = beta+B
    RGB = np.array([R,G,B])
    RGB = RGB/np.max(RGB)
    luminance = dot(LRGB,RGB)
    if luminance<l:
        a = (l-1)/(luminance-1)
        RGB = a*RGB + (1-a)*ones(3)
    elif luminance>l:
        RGB *= l/luminance
    return clip(RGB,0,1)

# Sphinx can't get past this static initializer code
try:
    __N_HL_LUT__ = 256
    __HL_LUT__ = np.zeros((__N_HL_LUT__,__N_HL_LUT__,3),dtype=np.float32)
    for ih,h in enum(np.linspace(0,2*3.141592653589793,__N_HL_LUT__+1)[:-1]):
        for il,l in enum(np.linspace(0,1,__N_HL_LUT__)):
            r,g,b = radl2rgb(h,l)
            __HL_LUT__[ih,il] = r,g,b
except:
    pass


def radl2rgbLUT(h,l=1.0):
    '''
    radl2rgb backed with a limited resolution lookup table
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    N = __N_HL_LUT__
    return __HL_LUT__[int(h*N/(2*pi))%N,int(l*(N-1))]


def complexHLArr2RGB(z):
    ''' 
    Performs bulk LUT for complex numbers, avoids loops
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    N = __N_HL_LUT__
    h = np.ravel(np.int32(np.angle(z)*N/(2*pi))%N)
    v = np.ravel(np.int32(np.clip(np.abs(z),0,1)*(N-1)))
    return np.reshape(__HL_LUT__[h,v],np.shape(z)+(3,))

######################################################################
# Matplotlib extensions

def color_boxplot(bp,COLOR):
    '''
    The Boxplot defaults are awful.
    This is a little better
    '''
    pylab.setp(bp['boxes'], color=COLOR, edgecolor=COLOR)
    pylab.setp(bp['whiskers'], color=COLOR, ls='-', lw=1)
    pylab.setp(bp['caps'], color=COLOR, lw=1)
    pylab.setp(bp['fliers'], color=COLOR, ms=4)
    pylab.setp(bp['medians'], color=GATHER[-1], lw=1.5, solid_capstyle='butt')

######################################################################
# Bit-packed color fiddling


def hex_pack_BGR(RGB):
    '''
    Packs RGB colors data in hexadecimal BGR format for fast rendering to
    Javascript canvas.

    This is useful for importing color-maps into HTML and javascript demos.

    Parameters
    ----------
    RGB: 256x3 array-like
        RGB values
    Returns
    -------
    list:
        List of HTML hex codes for the given RGB colors
    '''
    RGB = clip(np.array(RGB),0,1)
    return ['0x%02x%02x%02x'%(B*255,G*255,R*255) for (R,G,B) in RGB]


def code_to_16bit(code):
    '''
    Converts a #RRGGBB hex code into a 16-bit packed 565 BRG integer form
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    R = code[1:3]
    G = code[3:5]
    B = code[5:7]
    R = int(R,base=16)
    G = int(G,base=16)
    B = int(B,base=16)
    R = R & 0xF8
    G = G & 0xFC
    B = B & 0xF8
    return (R)|(G<<3)|(B>>3)


def bit16_RGB_to_tuple(RGB):
    '''
    Converts 16-bit RRRRR GGGGGG BBBBB into (R,G,B) tuple form
    
    Parameters
    ----------
    RGB: int
        16-bit RRRRR GGGGGG BBBBB packed color
    
    Returns
    -------
    R,G,B
        Components in [0,1]
    '''
    R = float(0b11111  & (RGB>>11))/0b11111
    G = float(0b111111 & (RGB>>5) )/0b111111
    B = float(0b11111  & (RGB)    )/0b11111
    return R,G,B


def enumerate_fast_colors():
    '''
    This is for the Arduino TFT touch screen shield from Adafruit.
    
    Enumerates colors that can be rendered over a 16 bit bus
    using two identical bytes, skipping the lowest two bits of
    each byte, and reserving the fourth bit for mask storage.
    This is intended for development of optimized color pallets for
    mictrocontroller driven display interfaces.
    
    Returns
    -------
    colors:
    '''
    bytes = sorted(list(set([i&0b11110100 for i in range(0,256)])))
    colors = [bit16_RGB_to_tuple(x*256|x) for x in bytes]
    return colors


def tuple_to_bit16(c):
    '''
    convert RGB float tuple in 565 bit packed RGB format
    
    Parameters
    ----------
    c : RGB color tuple
    
    Returns
    -------
    RGB:
    '''
    R,G,B = c
    R = int(R*0b11111)
    G = int(G*0b111111)
    B = int(B*0b11111)
    RGB = (R<<11)|(G<<5)|B
    return RGB


def tuple_to_bit24(c):
    '''
    convert RGB float tuple in 565 bit packed RGB format
    
    Parameters
    ----------
    c : RGB color tuple
    
    Returns
    -------
    '''
    R,G,B = c
    R = int(R*0b11111111)
    G = int(G*0b11111111)
    B = int(B*0b11111111)
    RGB = (R<<16)|(G<<8)|B
    return RGB


def bit16_print_color(c):
    '''
    Convert RGB tuple to 16 bit 565 RGB formt as a binary string literal
    
    Parameters
    ----------
    c : RGB color tuple
    
    Returns
    -------
    '''
    return bin(tuple_to_bit16(c))


def show_fast_pallet():
    '''
    Subset of colors that can be shownn quickly on the Arduino UNO using
    the standard 3.2-inch TFT touch-screen breakout. Restricts colors that
    can be sent with a single write to one port. 16-bit color mode with
    low and high bytes identical. 
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    plt.figure(figsize=(5,5),facecolor=(1,)*4)
    ax=plt.subplot(111)
    plt.subplots_adjust(0,0,1,1,0,0)
    colors = enumerate_fast_colors()
    i = 0;
    plt.xlim(0,4)
    plt.ylim(0,4)
    for ik,k in enumerate([0,3,5,7]):
        for j in range(4):
            i = k*4+j
            # We use one of the bits as a color flag; show the average color
            # with this bit set and unset. 
            c0 = colors[i]
            c2 = bit16_RGB_to_tuple(tuple_to_bit16(c0)|0b0000100000001000)
            cA = .5*(np.array(c0)+np.array(c2))
            ax.add_patch(matplotlib.patches.Rectangle((ik,j),1,1,facecolor=cA))
            print('%06x'%tuple_to_bit24(cA))
    plt.draw()


def show_complete_fast_pallet():
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    plt.figure(figsize=(10,5),facecolor=(1,)*4)
    ax=plt.subplot(111)
    plt.subplots_adjust(0,0,1,1,0,0)
    colors = enumerate_fast_colors()
    i = 0;
    plt.xlim(0,8)
    plt.ylim(0,4)
    for k in range(8):
        for j in range(4):
            i = k*4+j
            c0 = colors[i]
            c2 = bit16_RGB_to_tuple(tuple_to_bit16(c0)|0b0000100000001000)
            cA = .5*(np.array(c0)+np.array(c2))
            ki = k if k<4 else 4-k
            ji = j*2 if k<4 else j*2+1
            ax.add_patch(matplotlib.patches.Rectangle((ki,ji),1,1,facecolor=cA,edgecolor=cA))
            print('#%06x'%tuple_to_bit24(cA))
    plt.draw()


def show_complete_fastest_pallet():
    '''
    16 bit RGB 565; but don't even write the lower byte
    '''
    plt.figure(figsize=(10,5),facecolor=(1,)*4)
    ax=plt.subplot(111)
    plt.subplots_adjust(0,0,1,1,0,0)
    colors = enumerate_fast_colors()
    i = 0;
    plt.xlim(0,8)
    plt.ylim(0,4)
    for k in range(8):
        for ij,j in enumerate([0,2,1,3]):
            i = k*4+j
            c  = tuple_to_bit16(colors[i])
            c0 = bit16_RGB_to_tuple(c & 0b1111111100000000)
            c2 = bit16_RGB_to_tuple((c|0b0000100000001000)&0b1111111100000000)
            cA = .5*(np.array(c0)+np.array(c2))
            ax.add_patch(matplotlib.patches.Rectangle((k,ij),1,1,facecolor=cA,edgecolor=cA))
            print("'#%06x',"%tuple_to_bit24(cA),)
    plt.draw()


def show_hex_pallet(colors):
    '''
    Parameters
    ----------
    colors: list of colors to show

    '''
    plt.figure(figsize=(5,5),facecolor=(1,)*4)
    ax=plt.subplot(111)
    plt.subplots_adjust(0,0,1,1,0,0)
    N = int(np.ceil(np.sqrt(len(colors))))
    plt.xlim(0,N)
    plt.ylim(0,N)
    for i,c in enumerate(colors):
        x = i % N
        y = i // N
        ax.add_patch(matplotlib.patches.Rectangle((x,y),1,1,facecolor=c,edgecolor=c))

'''
show_hex_pallet([
'#040000', '#350000', '#560000',
'#770000',
'#870000', '#980000',
'#a80000', '#b90000',
'#c90000', '#d90000',
'#fa0000',
'#048100',
'#458100', '#568100',
'#668100', '#778100',
'#878100', '#988100',
'#a88100', '#b98100',
'#c98100', '#d98100',
'#fa8100',
])
#040000
#140000
#048100
#148100
#250000
#350000
#258100
#358100
#450000
#560000
#458100
#568100
#660000
#770000
#668100
#778100
#870000
#980000
#878100
#988100
#a80000
#b90000
#a88100
#b98100
#c90000
#d90000
#c98100
#d98100
#ea0000
#fa0000
#ea8100
#fa8100
#define FAST_BLACK   0b0b0000000000000000
#define FAST_GREEN   0b0b0000010000000100
#define FAST_BLUE    0b0b0001000000010000
#define FAST_CYAN    0b0b0001010000010100
#define FAST_CRIMSON 0b0b1000000010000000
#define FAST_OLIVE   0b0b1000010010000100
#define FAST_VIOLET  0b0b1001000010010000
#define FAST_GREY    0b0b1001010010010100
#define FAST_RED     0b0b1110000011100000
#define FAST_YELLOW  0b0b1110010011100100
#define FAST_MAGENTA 0b0b1111000011110000
#define FAST_WHITE   0b0b1111010011110100
'''


