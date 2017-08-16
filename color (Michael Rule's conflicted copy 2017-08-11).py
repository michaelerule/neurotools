#!/usr/bin/python
# -*- coding: UTF-8 -*-
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

'''
Miscellaneous color-related functions. Several color maps for use with
matplotlib. A couple idiosyncratic color pallets.

Defines the color maps parula, isolum, and extended

This class also defines three hue wheel color maps of varying brightness

>>> lighthue = mcolors.ListedColormap(lighthues(NCMAP),'lighthue')
>>> medhue   = mcolors.ListedColormap(medhues  (NCMAP),'medhue')
>>> darkhue  = mcolors.ListedColormap(darkhues (NCMAP),'darkhue')
'''

import math
import pylab
from neurotools import signal
from neurotools import tools
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os.path    import expanduser
from matplotlib import cm
from numpy import pi,e
from neurotools.signal.signal import gaussian_smooth

# This is the color scheme from the painting "gather" by bridget riley
GATHER = [
'#f1f0e9', # "White"
'#eb7a59', # "Rust"
'#eea300', # "Sand"
'#5aa0df', # "Azure"
'#00bac9', # "Turquoise"
'#44525c'] # "Black"
WHITE,RUST,OCHRE,AZURE,TURQUOISE,BLACK = GATHER

# completes the Gather spectrum
MOSS  = '#77ae64'
MAUVE = '#956f9b'
INDEGO     = [.37843053,  .4296282 ,  .76422011]
VERIDIAN   = [.06695279,  .74361409,  .55425139]
CHARTREUSE = [.71152929,  .62526339,  .10289384]
CRIMSON    = [.84309675,  .37806273,  .32147779]

######################################################################
# Hue / Saturation / Luminance color space code

def hsv2rgb(h,s,v):
    '''
    h: hue 0 to 360
    s: saturation 0 to 1
    v: value 0 to 1
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
    return ((v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q))[hi]

def lightness(r,g,b,method='lightness'):
    return luminance_matrix(method=method).dot((r,g,b))

def luminance_matrix(method='perceived'):
    '''
    Method 'perceived' .299*R + .587*G + .114*B
    Method 'standard'  .2126*R + .7152*G + .0722*B
    Method 'lightness' .3*R + .59*G + .11*B
    '''
    if method=='standard':
        x1 = .2126
        x2 = .7152
        x3 = .0722
    elif method=='perceived':
        x1 = .299
        x2 = .587
        x3 = .114
    elif method=='lightness':
        x1 = .30
        x2 = .59
        x3 = .11
    else:
        assert 0
    LRGB = np.array([x1,x2,x3])
    LRGB = LRGB / np.sum(LRGB)
    return LRGB

def match_luminance(target,color,
    THRESHOLD=0.01,squared=False,method='perceived'):
    '''
    Adjust color to match luminosity of target

    Method 'perceived' .299*R + .587*G + .114*B
    Method 'standard'  .2126*R + .7152*G + .0722*B
    Method 'lightness' .3*R + .59*G + .11*B
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
    # do a bounded relaxation of the color to attempt to tweak luminance
    # while preserving hue, saturation as much as is possible
    while abs(source-luminance)>THRESHOLD:
        arithmetic_corrected = clip(color+luminance-source,0,1)
        geometric_corrected  = clip(color*luminance/source,0,1)
        correction = .5*(arithmetic_corrected+geometric_corrected)
        color = .9*color+0.1*np.clip(correction,0,1)
        if squared:
            source    = np.dot(LRGB,color**2)**0.5
        else:
            source    = np.dot(LRGB,color)
    return color

def rotate(colors,th):
    '''
    Rotate a list of rgb colors by angle theta
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
    alpha  = .5*(2*r-g-b)
    beta   = sqrt(3)/2*(g-b)
    hue    = arctan2(beta,alpha)
    chroma = sqrt(alpha**2+beta**2)
    L = lightness(r,g,b)
    return hue,chroma,L

def hue_angle(c1,c2):
    '''
    Calculates the angular difference in hue between two colors
    '''
    H1 = RGBtoHCL(*c1)[0]
    H2 = RGBtoHCL(*c2)[0]
    return H2-H1

def hcl2rgb(h,c,l,target = 1.0, method='standard'):
    '''
    h: hue
    c: chroma
    l: luminosity
    '''
    if method=='standard':
        x1 = .2126
        x2 = .7152
        x3 = .0722
    elif method=='perceived':
        x1 = .299
        x2 = .587
        x3 = .114
    elif method=='lightness':
        x1 = .30
        x2 = .59
        x3 = .11
    LRGB = luminance_matrix()
    h = h*pi/180.0
    alpha = np.cos(h)
    beta = np.sin(h)*2/np.sqrt(3)
    B = l - x1*(alpha+beta/2)-x2*beta
    R = alpha + beta/2+B
    G = beta+B
    RGB = np.array([R,G,B])
    return np.clip(RGB,0,1)

def circularly_smooth_colormap(cm,s):
    '''
    Smooth a colormap with cirular boundary conditions

    s: sigma, standard dev of gaussian smoothing kernel in samples
    cm: color map, array-like of RGB tuples
    '''
    # Do circular boundary conditions the lazy way
    cm = np.array(cm)
    N = cm.shape[0]
    cm = np.concatenate([cm,cm,cm])
    R,G,B = cm.T
    R = gaussian_smooth(R,s)
    G = gaussian_smooth(G,s)
    B = gaussian_smooth(B,s)
    RGB = np.array([R,G,B]).T
    return RGB[N:N*2,:]

def isoluminance1(h,l=.5):
    return hcl2rgb(h,1,1,target=float(l))

def isoluminance2(h):
    return hcl2rgb(h,1,1.0,target=.5)*(1+(h%5))/5

def isoluminance3(h):
    return hcl2rgb(h,1,1.0,target=.5)*(1+(h%15))/15

def isoluminance4(h):
    return hcl2rgb(h,1,1.0,target=.5)*(1+(h%60))/60

def lighthues(N=10,l=0.7):
    return [isoluminance1(h,l) for h in np.linspace(0,360,N+1)[:-1]]

def darkhues(N=10,l=0.4):
    return [isoluminance1(h,l) for h in np.linspace(0,360,N+1)[:-1]]

def medhues(N=10,l=0.6):
    return [isoluminance1(h,l) for h in np.linspace(0,360,N+1)[:-1]]

def radl2rgb(h,l=1.0):
    '''
    Slightly more optimized HSL conversion routine.
    Saturation fixed at 1
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

# this is a problem. Grr.
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
    '''
    N = __N_HL_LUT__
    return __HL_LUT__[int(h*N/(2*pi))%N,int(l*(N-1))]

def complexHLArr2RGB(z):
    ''' Performs bulk LUT for complex numbers, avoids loops'''
    N = __N_HL_LUT__
    h = np.ravel(np.int32(np.angle(z)*N/(2*pi))%N)
    v = np.ravel(np.int32(np.clip(np.abs(z),0,1)*(N-1)))
    return np.reshape(__HL_LUT__[h,v],shape(z)+(3,))




####################################################################### Matplotlib extensions

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


####################################################################### Three isoluminance hue wheels at varying brightness
# Unfortunately the hue distribution is a bit off for these and they come
# out a little heavy in the red, gree, and blue. I don't reccommend using
# them
try:
    lighthue = mcolors.ListedColormap(lighthues(360),'lighthue')
    medhue   = mcolors.ListedColormap(medhues  (360),'medhue')
    darkhue  = mcolors.ListedColormap(darkhues (360),'darkhue')
except:
    pass
    #sphinx was crashing

####################################################################### Isoluminance hue wheel color map
# pip install husl
# http://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/28790/versions/5/screenshot.jpg
# https://pypi.python.org/pypi/husl
# https://mycarta.wordpress.com/2014/10/30/new-matlab-isoluminant-colormap-for-azimuth-data/
# radius = 38; % chroma
# theta = linspace(0, 2*pi, 256)'; % hue
# a = radius * np.cos(theta);
# b = radius * np.sin(theta);
# L = (ones(1, 256)*65)'; % lightness
# Lab = [L, a, b];
# RGB=colorspace('RGB<-Lab',Lab(end:-1:1,:));
# https://mycarta.wordpress.com/
isolum_data = [
[.8658,0.5133,0.6237],[.8638,0.5137,0.6301],[.8615,0.5141,0.6366],
[.8591,0.5147,0.6430],[.8566,0.5153,0.6494],[.8538,0.5160,0.6557],
[.8509,0.5168,0.6621],[.8479,0.5176,0.6684],[.8446,0.5185,0.6747],
[.8412,0.5195,0.6810],[.8376,0.5206,0.6872],[.8338,0.5218,0.6934],
[.8299,0.5230,0.6996],[.8258,0.5243,0.7057],[.8215,0.5257,0.7118],
[.8171,0.5271,0.7178],[.8125,0.5286,0.7237],[.8077,0.5302,0.7296],
[.8028,0.5318,0.7355],[.7977,0.5335,0.7412],[.7924,0.5352,0.7469],
[.7870,0.5370,0.7525],[.7814,0.5389,0.7581],[.7756,0.5408,0.7635],
[.7697,0.5427,0.7689],[.7636,0.5447,0.7742],[.7573,0.5467,0.7794],
[.7509,0.5488,0.7845],[.7443,0.5509,0.7895],[.7376,0.5530,0.7944],
[.7307,0.5552,0.7992],[.7237,0.5574,0.8039],[.7165,0.5596,0.8085],
[.7092,0.5618,0.8130],[.7017,0.5641,0.8173],[.6940,0.5664,0.8216],
[.6862,0.5687,0.8257],[.6783,0.5710,0.8297],[.6702,0.5733,0.8336],
[.6620,0.5757,0.8373],[.6536,0.5780,0.8409],[.6451,0.5804,0.8444],
[.6365,0.5828,0.8477],[.6277,0.5851,0.8509],[.6188,0.5875,0.8540],
[.6097,0.5898,0.8569],[.6005,0.5922,0.8597],[.5912,0.5946,0.8623],
[.5818,0.5969,0.8648],[.5722,0.5992,0.8672],[.5625,0.6016,0.8693],
[.5527,0.6039,0.8714],[.5427,0.6062,0.8733],[.5326,0.6085,0.8750],
[.5224,0.6107,0.8766],[.5121,0.6130,0.8780],[.5017,0.6152,0.8792],
[.4911,0.6174,0.8803],[.4804,0.6196,0.8813],[.4696,0.6218,0.8820],
[.4587,0.6239,0.8826],[.4476,0.6260,0.8831],[.4365,0.6281,0.8834],
[.4252,0.6302,0.8835],[.4138,0.6322,0.8835],[.4022,0.6342,0.8833],
[.3905,0.6362,0.8829],[.3787,0.6381,0.8824],[.3668,0.6400,0.8817],
[.3547,0.6419,0.8809],[.3425,0.6437,0.8799],[.3301,0.6455,0.8787],
[.3175,0.6473,0.8774],[.3048,0.6491,0.8759],[.2918,0.6508,0.8742],
[.2787,0.6524,0.8724],[.2653,0.6541,0.8705],[.2517,0.6557,0.8684],
[.2377,0.6572,0.8661],[.2235,0.6588,0.8637],[.2088,0.6602,0.8611],
[.1937,0.6617,0.8584],[.1781,0.6631,0.8556],[.1617,0.6645,0.8526],
[.1444,0.6658,0.8494],[.1260,0.6671,0.8461],[.1059,0.6684,0.8427],
[.0832,0.6696,0.8391],[.0560,0.6708,0.8354],[.0206,0.6720,0.8316],
[.0000,0.6739,0.8282],[.0000,0.6767,0.8255],[.0000,0.6794,0.8226],
[.0000,0.6819,0.8194],[.0000,0.6841,0.8161],[.0000,0.6862,0.8124],
[.0000,0.6880,0.8086],[.0000,0.6897,0.8045],[.0000,0.6912,0.8003],
[.0000,0.6925,0.7958],[.0000,0.6936,0.7911],[.0000,0.6946,0.7862],
[.0000,0.6953,0.7810],[.0000,0.6959,0.7757],[.0000,0.6964,0.7702],
[.0000,0.6966,0.7645],[.0000,0.6968,0.7586],[.0000,0.6967,0.7525],
[.0000,0.6965,0.7461],[.0000,0.6962,0.7396],[.0000,0.6957,0.7330],
[.0000,0.6950,0.7261],[.0000,0.6942,0.7190],[.0000,0.6933,0.7118],
[.0000,0.6922,0.7043],[.0000,0.6909,0.6967],[.0000,0.6896,0.6889],
[.0221,0.6893,0.6821],[.0525,0.6894,0.6757],[.0765,0.6895,0.6693],
[.0965,0.6895,0.6629],[.1140,0.6895,0.6564],[.1300,0.6895,0.6499],
[.1448,0.6894,0.6434],[.1588,0.6893,0.6369],[.1720,0.6892,0.6304],
[.1847,0.6890,0.6238],[.1968,0.6889,0.6173],[.2086,0.6886,0.6107],
[.2200,0.6884,0.6042],[.2311,0.6881,0.5976],[.2419,0.6878,0.5911],
[.2524,0.6874,0.5846],[.2627,0.6870,0.5780],[.2728,0.6866,0.5716],
[.2827,0.6862,0.5651],[.2925,0.6857,0.5587],[.3020,0.6852,0.5523],
[.3115,0.6846,0.5459],[.3207,0.6841,0.5396],[.3299,0.6835,0.5333],
[.3389,0.6828,0.5271],[.3478,0.6821,0.5209],[.3566,0.6814,0.5147],
[.3653,0.6807,0.5087],[.3739,0.6799,0.5026],[.3824,0.6791,0.4967],
[.3908,0.6783,0.4908],[.3992,0.6774,0.4850],[.4074,0.6765,0.4793],
[.4156,0.6756,0.4736],[.4237,0.6746,0.4680],[.4317,0.6736,0.4625],
[.4397,0.6726,0.4571],[.4476,0.6716,0.4518],[.4554,0.6705,0.4466],
[.4632,0.6693,0.4415],[.4709,0.6682,0.4365],[.4785,0.6670,0.4316],
[.4861,0.6658,0.4268],[.4937,0.6645,0.4221],[.5012,0.6633,0.4176],
[.5086,0.6619,0.4131],[.5160,0.6606,0.4088],[.5234,0.6592,0.4046],
[.5306,0.6578,0.4005],[.5379,0.6564,0.3966],[.5451,0.6549,0.3928],
[.5522,0.6534,0.3891],[.5593,0.6519,0.3856],[.5664,0.6503,0.3822],
[.5734,0.6487,0.3790],[.5803,0.6471,0.3759],[.5872,0.6454,0.3730],
[.5941,0.6437,0.3702],[.6009,0.6420,0.3676],[.6077,0.6403,0.3652],
[.6144,0.6385,0.3629],[.6211,0.6367,0.3607],[.6277,0.6349,0.3588],
[.6342,0.6331,0.3570],[.6407,0.6312,0.3553],[.6472,0.6293,0.3539],
[.6536,0.6273,0.3526],[.6600,0.6254,0.3515],[.6662,0.6234,0.3505],
[.6725,0.6214,0.3498],[.6787,0.6194,0.3492],[.6848,0.6174,0.3488],
[.6908,0.6153,0.3485],[.6968,0.6132,0.3485],[.7027,0.6111,0.3486],
[.7086,0.6090,0.3489],[.7144,0.6069,0.3493],[.7201,0.6047,0.3500],
[.7258,0.6026,0.3508],[.7314,0.6004,0.3518],[.7369,0.5982,0.3529],
[.7423,0.5960,0.3542],[.7477,0.5938,0.3557],[.7529,0.5916,0.3574],
[.7581,0.5893,0.3592],[.7633,0.5871,0.3612],[.7683,0.5849,0.3633],
[.7732,0.5827,0.3656],[.7781,0.5804,0.3680],[.7828,0.5782,0.3706],
[.7875,0.5760,0.3734],[.7921,0.5737,0.3763],[.7966,0.5715,0.3793],
[.8010,0.5693,0.3825],[.8052,0.5671,0.3858],[.8094,0.5649,0.3892],
[.8135,0.5628,0.3928],[.8175,0.5606,0.3965],[.8213,0.5585,0.4004],
[.8251,0.5564,0.4043],[.8287,0.5543,0.4084],[.8322,0.5522,0.4126],
[.8356,0.5502,0.4169],[.8389,0.5482,0.4214],[.8421,0.5462,0.4259],
[.8451,0.5443,0.4306],[.8481,0.5424,0.4353],[.8509,0.5405,0.4402],
[.8535,0.5387,0.4451],[.8561,0.5369,0.4502],[.8585,0.5352,0.4553],
[.8607,0.5335,0.4605],[.8629,0.5318,0.4658],[.8649,0.5303,0.4712],
[.8667,0.5287,0.4767],[.8685,0.5273,0.4822],[.8700,0.5259,0.4879],
[.8715,0.5245,0.4936],[.8728,0.5232,0.4993],[.8739,0.5220,0.5051],
[.8749,0.5209,0.5110],[.8757,0.5198,0.5170],[.8764,0.5188,0.5230],
[.8770,0.5178,0.5290],[.8773,0.5170,0.5351],[.8776,0.5162,0.5413],
[.8776,0.5155,0.5475],[.8775,0.5148,0.5537],[.8773,0.5143,0.5599],
[.8769,0.5138,0.5662],[.8763,0.5134,0.5725],[.8756,0.5131,0.5789],
[.8747,0.5129,0.5853],[.8736,0.5128,0.5916],[.8724,0.5127,0.5980],
[.8710,0.5127,0.6045],[.8695,0.5128,0.6109],[.8677,0.5130,0.6173],
[.8658,0.5133,0.6237]]

try:
    isolum = mcolors.ListedColormap(isolum_data,'isolum')
    plt.register_cmap(name='isolum', cmap=isolum)
    isolum_data = np.float32(isolum_data)
    double_isolum_data = np.concatenate(
        [isolum_data[::2],isolum_data[::2]])
    double_isolum = mcolors.ListedColormap(
        double_isolum_data,'isolum')
    plt.register_cmap(name='double_isolum', cmap=double_isolum)
except:
    pass
    #sphinx workaround



####################################################################### Parula color map
parula_data = [
[.2081, .1663, .5292],[.2091, .1721, .5411],[.2101, .1779, .5530],
[.2109, .1837, .5650],[.2116, .1895, .5771],[.2121, .1954, .5892],
[.2124, .2013, .6013],[.2125, .2072, .6135],[.2123, .2132, .6258],
[.2118, .2192, .6381],[.2111, .2253, .6505],[.2099, .2315, .6629],
[.2084, .2377, .6753],[.2063, .2440, .6878],[.2038, .2503, .7003],
[.2006, .2568, .7129],[.1968, .2632, .7255],[.1921, .2698, .7381],
[.1867, .2764, .7507],[.1802, .2832, .7634],[.1728, .2902, .7762],
[.1641, .2975, .7890],[.1541, .3052, .8017],[.1427, .3132, .8145],
[.1295, .3217, .8269],[.1147, .3306, .8387],[.0986, .3397, .8495],
[.0816, .3486, .8588],[.0646, .3572, .8664],[.0482, .3651, .8722],
[.0329, .3724, .8765],[.0213, .3792, .8796],[.0136, .3853, .8815],
[.0086, .3911, .8827],[.0060, .3965, .8833],[.0051, .4017, .8834],
[.0054, .4066, .8831],[.0067, .4113, .8825],[.0089, .4159, .8816],
[.0116, .4203, .8805],[.0148, .4246, .8793],[.0184, .4288, .8779],
[.0223, .4329, .8763],[.0264, .4370, .8747],[.0306, .4410, .8729],
[.0349, .4449, .8711],[.0394, .4488, .8692],[.0437, .4526, .8672],
[.0477, .4564, .8652],[.0514, .4602, .8632],[.0549, .4640, .8611],
[.0582, .4677, .8589],[.0612, .4714, .8568],[.0640, .4751, .8546],
[.0666, .4788, .8525],[.0689, .4825, .8503],[.0710, .4862, .8481],
[.0729, .4899, .8460],[.0746, .4937, .8439],[.0761, .4974, .8418],
[.0773, .5012, .8398],[.0782, .5051, .8378],[.0789, .5089, .8359],
[.0794, .5129, .8341],[.0795, .5169, .8324],[.0793, .5210, .8308],
[.0788, .5251, .8293],[.0778, .5295, .8280],[.0764, .5339, .8270],
[.0746, .5384, .8261],[.0724, .5431, .8253],[.0698, .5479, .8247],
[.0668, .5527, .8243],[.0636, .5577, .8239],[.0600, .5627, .8237],
[.0562, .5677, .8234],[.0523, .5727, .8231],[.0484, .5777, .8228],
[.0445, .5826, .8223],[.0408, .5874, .8217],[.0372, .5922, .8209],
[.0342, .5968, .8198],[.0317, .6012, .8186],[.0296, .6055, .8171],
[.0279, .6097, .8154],[.0265, .6137, .8135],[.0255, .6176, .8114],
[.0248, .6214, .8091],[.0243, .6250, .8066],[.0239, .6285, .8039],
[.0237, .6319, .8010],[.0235, .6352, .7980],[.0233, .6384, .7948],
[.0231, .6415, .7916],[.0230, .6445, .7881],[.0229, .6474, .7846],
[.0227, .6503, .7810],[.0227, .6531, .7773],[.0232, .6558, .7735],
[.0238, .6585, .7696],[.0246, .6611, .7656],[.0263, .6637, .7615],
[.0282, .6663, .7574],[.0306, .6688, .7532],[.0338, .6712, .7490],
[.0373, .6737, .7446],[.0418, .6761, .7402],[.0467, .6784, .7358],
[.0516, .6808, .7313],[.0574, .6831, .7267],[.0629, .6854, .7221],
[.0692, .6877, .7173],[.0755, .6899, .7126],[.0820, .6921, .7078],
[.0889, .6943, .7029],[.0956, .6965, .6979],[.1031, .6986, .6929],
[.1104, .7007, .6878],[.1180, .7028, .6827],[.1258, .7049, .6775],
[.1335, .7069, .6723],[.1418, .7089, .6669],[.1499, .7109, .6616],
[.1585, .7129, .6561],[.1671, .7148, .6507],[.1758, .7168, .6451],
[.1849, .7186, .6395],[.1938, .7205, .6338],[.2033, .7223, .6281],
[.2128, .7241, .6223],[.2224, .7259, .6165],[.2324, .7275, .6107],
[.2423, .7292, .6048],[.2527, .7308, .5988],[.2631, .7324, .5929],
[.2735, .7339, .5869],[.2845, .7354, .5809],[.2953, .7368, .5749],
[.3064, .7381, .5689],[.3177, .7394, .5630],[.3289, .7406, .5570],
[.3405, .7417, .5512],[.3520, .7428, .5453],[.3635, .7438, .5396],
[.3753, .7446, .5339],[.3869, .7454, .5283],[.3986, .7461, .5229],
[.4103, .7467, .5175],[.4218, .7473, .5123],[.4334, .7477, .5072],
[.4447, .7482, .5021],[.4561, .7485, .4972],[.4672, .7487, .4924],
[.4783, .7489, .4877],[.4892, .7491, .4831],[.5000, .7491, .4786],
[.5106, .7492, .4741],[.5212, .7492, .4698],[.5315, .7491, .4655],
[.5418, .7490, .4613],[.5519, .7489, .4571],[.5619, .7487, .4531],
[.5718, .7485, .4490],[.5816, .7482, .4451],[.5913, .7479, .4412],
[.6009, .7476, .4374],[.6103, .7473, .4335],[.6197, .7469, .4298],
[.6290, .7465, .4261],[.6382, .7460, .4224],[.6473, .7456, .4188],
[.6564, .7451, .4152],[.6653, .7446, .4116],[.6742, .7441, .4081],
[.6830, .7435, .4046],[.6918, .7430, .4011],[.7004, .7424, .3976],
[.7091, .7418, .3942],[.7176, .7412, .3908],[.7261, .7405, .3874],
[.7346, .7399, .3840],[.7430, .7392, .3806],[.7513, .7385, .3773],
[.7596, .7378, .3739],[.7679, .7372, .3706],[.7761, .7364, .3673],
[.7843, .7357, .3639],[.7924, .7350, .3606],[.8005, .7343, .3573],
[.8085, .7336, .3539],[.8166, .7329, .3506],[.8246, .7322, .3472],
[.8325, .7315, .3438],[.8405, .7308, .3404],[.8484, .7301, .3370],
[.8563, .7294, .3336],[.8642, .7288, .3300],[.8720, .7282, .3265],
[.8798, .7276, .3229],[.8877, .7271, .3193],[.8954, .7266, .3156],
[.9032, .7262, .3117],[.9110, .7259, .3078],[.9187, .7256, .3038],
[.9264, .7256, .2996],[.9341, .7256, .2953],[.9417, .7259, .2907],
[.9493, .7264, .2859],[.9567, .7273, .2808],[.9639, .7285, .2754],
[.9708, .7303, .2696],[.9773, .7326, .2634],[.9831, .7355, .2570],
[.9882, .7390, .2504],[.9922, .7431, .2437],[.9952, .7476, .2373],
[.9973, .7524, .2310],[.9986, .7573, .2251],[.9991, .7624, .2195],
[.9990, .7675, .2141],[.9985, .7726, .2090],[.9976, .7778, .2042],
[.9964, .7829, .1995],[.9950, .7880, .1949],[.9933, .7931, .1905],
[.9914, .7981, .1863],[.9894, .8032, .1821],[.9873, .8083, .1780],
[.9851, .8133, .1740],[.9828, .8184, .1700],[.9805, .8235, .1661],
[.9782, .8286, .1622],[.9759, .8337, .1583],[.9736, .8389, .1544],
[.9713, .8441, .1505],[.9692, .8494, .1465],[.9672, .8548, .1425],
[.9654, .8603, .1385],[.9638, .8659, .1343],[.9623, .8716, .1301],
[.9611, .8774, .1258],[.9600, .8834, .1215],[.9593, .8895, .1171],
[.9588, .8958, .1126],[.9586, .9022, .1082],[.9587, .9088, .1036],
[.9591, .9155, .0990],[.9599, .9225, .0944],[.9610, .9296, .0897],
[.9624, .9368, .0850],[.9641, .9443, .0802],[.9662, .9518, .0753],
[.9685, .9595, .0703],[.9710, .9673, .0651],[.9736, .9752, .0597],
[.9763, .9831, .0538]]

try:
    parula = mcolors.ListedColormap(parula_data,'parula')
    plt.register_cmap(name='parula', cmap=parula)
    parula_data = np.float32(parula_data)
except:
    parula=None
    pass
    #sphinx

#############################################################################
# A parula-like color map, extended all the way to black at one end
extended_data = np.array([
[ 11,   0,   0],[ 12,   0,   0],[ 15,   0,   0],[ 17,   0,   1],
[ 19,   0,   1],[ 21,   0,   2],[ 23,   1,   2],[ 26,   1,   3],
[ 28,   1,   3],[ 30,   1,   4],[ 32,   1,   4],[ 34,   1,   4],
[ 36,   2,   5],[ 39,   2,   5],[ 41,   2,   6],[ 43,   2,   6],
[ 45,   2,   7],[ 47,   3,   7],[ 49,   3,  10],[ 50,   4,  13],
[ 51,   4,  15],[ 53,   5,  18],[ 54,   5,  21],[ 55,   6,  23],
[ 57,   6,  26],[ 58,   7,  29],[ 59,   7,  31],[ 61,   8,  34],
[ 62,   8,  36],[ 63,   9,  39],[ 65,   9,  42],[ 66,  10,  44],
[ 67,  10,  47],[ 69,  11,  50],[ 70,  11,  53],[ 69,  13,  58],
[ 67,  14,  64],[ 66,  15,  69],[ 64,  17,  75],[ 63,  18,  81],
[ 62,  20,  86],[ 60,  21,  92],[ 59,  22,  97],[ 57,  24, 103],
[ 56,  25, 109],[ 54,  27, 114],[ 53,  28, 120],[ 51,  29, 125],
[ 50,  31, 131],[ 48,  32, 137],[ 47,  33, 142],[ 46,  35, 148],
[ 44,  37, 152],[ 43,  38, 155],[ 41,  40, 159],[ 40,  42, 162],
[ 38,  44, 166],[ 37,  46, 169],[ 35,  47, 173],[ 34,  49, 177],
[ 32,  51, 180],[ 31,  53, 184],[ 29,  54, 187],[ 28,  56, 191],
[ 26,  58, 194],[ 25,  60, 198],[ 23,  62, 201],[ 22,  63, 205],
[ 20,  65, 208],[ 20,  67, 208],[ 19,  69, 207],[ 19,  71, 207],
[ 18,  74, 206],[ 18,  76, 205],[ 17,  78, 204],[ 17,  80, 203],
[ 16,  82, 203],[ 16,  84, 202],[ 15,  86, 201],[ 15,  88, 200],
[ 15,  90, 199],[ 14,  92, 199],[ 14,  94, 198],[ 13,  96, 197],
[ 13,  99, 196],[ 12, 101, 195],[ 12, 103, 194],[ 12, 104, 192],
[ 13, 106, 191],[ 13, 108, 189],[ 13, 110, 187],[ 13, 112, 186],
[ 13, 114, 184],[ 13, 116, 182],[ 14, 118, 181],[ 14, 120, 179],
[ 14, 122, 177],[ 14, 124, 176],[ 14, 126, 174],[ 14, 128, 172],
[ 15, 129, 171],[ 15, 131, 169],[ 15, 133, 167],[ 15, 135, 166],
[ 16, 137, 165],[ 16, 138, 164],[ 17, 140, 163],[ 17, 141, 162],
[ 18, 143, 161],[ 18, 145, 160],[ 19, 146, 159],[ 19, 148, 158],
[ 20, 150, 157],[ 20, 151, 156],[ 21, 153, 155],[ 21, 154, 154],
[ 22, 156, 153],[ 22, 158, 152],[ 23, 159, 151],[ 23, 161, 150],
[ 24, 163, 149],[ 24, 164, 147],[ 25, 166, 146],[ 25, 168, 145],
[ 26, 169, 143],[ 26, 171, 142],[ 27, 173, 141],[ 27, 175, 139],
[ 28, 176, 138],[ 28, 178, 137],[ 29, 180, 135],[ 29, 181, 134],
[ 30, 183, 133],[ 30, 185, 131],[ 31, 186, 130],[ 31, 188, 129],
[ 32, 190, 127],[ 33, 191, 126],[ 35, 192, 125],[ 37, 193, 125],
[ 38, 194, 124],[ 40, 195, 123],[ 41, 196, 122],[ 43, 197, 121],
[ 45, 198, 120],[ 46, 199, 119],[ 48, 200, 118],[ 49, 201, 117],
[ 51, 202, 117],[ 52, 203, 116],[ 54, 204, 115],[ 56, 206, 114],
[ 57, 207, 113],[ 59, 208, 112],[ 62, 208, 112],[ 65, 208, 111],
[ 68, 208, 111],[ 71, 209, 110],[ 74, 209, 110],[ 77, 209, 109],
[ 80, 210, 108],[ 83, 210, 108],[ 86, 210, 107],[ 89, 210, 107],
[ 92, 211, 106],[ 95, 211, 106],[ 98, 211, 105],[101, 211, 105],
[104, 212, 104],[107, 212, 104],[110, 212, 103],[113, 212, 102],
[117, 212, 101],[120, 212, 100],[124, 213,  99],[127, 213,  98],
[131, 213,  97],[134, 213,  96],[138, 213,  95],[141, 213,  95],
[145, 213,  94],[148, 213,  93],[152, 213,  92],[155, 213,  91],
[159, 214,  90],[162, 214,  89],[166, 214,  88],[169, 214,  87],
[173, 214,  87],[176, 214,  87],[179, 214,  87],[182, 214,  87],
[185, 214,  87],[189, 214,  87],[192, 214,  87],[195, 214,  87],
[198, 214,  87],[201, 214,  87],[205, 214,  87],[208, 215,  87],
[211, 215,  88],[214, 215,  88],[217, 215,  88],[221, 215,  88],
[224, 215,  88],[225, 215,  89],[227, 216,  90],[228, 217,  91],
[230, 218,  92],[232, 218,  93],[233, 219,  94],[235, 220,  96],
[236, 220,  97],[238, 221,  98],[239, 222,  99],[241, 222, 100],
[243, 223, 101],[244, 224, 102],[246, 224, 103],[247, 225, 105],
[249, 226, 106],[250, 226, 107],[251, 228, 108],[251, 229, 109],
[251, 230, 110],[251, 232, 111],[252, 233, 113],[252, 234, 114],
[252, 236, 115],[252, 237, 116],[253, 238, 117],[253, 240, 118],
[253, 241, 120],[253, 242, 121],[254, 244, 122],[254, 245, 123],
[254, 246, 124],[254, 248, 125],[254, 249, 127],[255, 249, 128],
[254, 250, 129],[255, 250, 131],[255, 250, 132],[254, 251, 133],
[255, 251, 135],[254, 251, 136],[254, 252, 137],[255, 252, 139],
[254, 252, 140],[255, 253, 141],[254, 253, 143],[254, 253, 144],
[255, 254, 145],[254, 254, 147],[255, 254, 148],[254, 254, 149]])
#try:
extended_data = np.float32(extended_data)/255
extended = mcolors.ListedColormap(extended_data,'extended')
plt.register_cmap(name='extended', cmap=extended)

##################################################################
# A balanced colormap that combines aspects of HSV, HSL, and
# Matteo Niccoli's isoluminant hue wheel, with a little smoothing
#
x = np.array([hsv2rgb(h,1,1) for h in np.linspace(0,1,256)*360])
y = np.array([hcl2rgb(h,1,0.75) for h in np.linspace(0,1,256)*360])
z = isolum_data[::-1]
balance_data = circularly_smooth_colormap(x*0.2+y*0.4+z*0.4,30)
balance = mcolors.ListedColormap(balance_data,'extended')
plt.register_cmap(name='balance', cmap=balance)
'''
except:
    extended=None
    isolum=None
    balance=None
    pass
    #sphinx bug workaround
'''
######################################################################
# Bit-packed color fiddling

def hex_pack_BGR(RGB):
    '''
    Packs RGB colors data in hexadecimal BGR format for fast rendering to
    Javascript canvas.
    RGB: 256x3 RGB array-like, ..1 values
    '''
    RGB = clip(np.array(RGB),0,1)
    return ['0x%2x%2x%2x'%(B*255,G*255,R*255) for (R,G,B) in RGB]

def code_to_16bit(code):
    '''
    Converts a #RRGGBB hex code into a 16-bit packed 565 BRG integer form
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
    '''
    R = float(0b11111  & (RGB>>11))/0b11111
    G = float(0b111111 & (RGB>>5) )/0b111111
    B = float(0b11111  & (RGB)    )/0b11111
    return R,G,B

def enumerate_fast_colors():
    '''
    Enumerates colors that can be rendered over a 16 bit bus
    using two identical bytes, skipping the lowest two bits of
    each byte, and reserving the fourth bit for mask storage.
    This is intended for development of optimized color pallets for
    mictrocontroller driven display interfaces.
    '''
    bytes = sorted(list(set([i&0b11110100 for i in range(0,256)])))
    colors = [bit16_RGB_to_tuple(x*256|x) for x in bytes]
    return colors

def tuple_to_bit16(c):
    '''
    convert RGB float tuple in 565 bit packed RGB format
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
    '''
    return bin(tuple_to_bit16(c))

def show_fast_pallet():
    figure(figsize=(5,5),facecolor=(1,)*4)
    ax=subplot(111)
    subplots_adjust(0,0,1,1,0,0)
    colors = enumerate_fast_colors()
    i = 0;
    xlim(0,3)
    ylim(0,4)
    for ik,k in enumerate([0,4,7]):
        for j in range(4):
            i = k*4+j
            c0 = colors[i]
            c2 = bit16_RGB_to_tuple(tuple_to_bit16(c0)|0b0000100000001000)
            cA = .5*(np.array(c0)+np.array(c2))
            ax.add_patch(patches.Rectangle((ik,j),1,1,facecolor=cA))
            print('%06x'%tuple_to_bit24(cA))
    draw()

def show_complete_fast_pallet():
    figure(figsize=(10,5),facecolor=(1,)*4)
    ax=subplot(111)
    subplots_adjust(0,0,1,1,0,0)
    colors = enumerate_fast_colors()
    i = 0;
    xlim(0,8)
    ylim(0,4)
    for k in range(8):
        for j in range(4):
            i = k*4+j
            c0 = colors[i]
            c2 = bit16_RGB_to_tuple(tuple_to_bit16(c0)|0b0000100000001000)
            cA = .5*(np.array(c0)+np.array(c2))
            ax.add_patch(patches.Rectangle((k,j),1,1,facecolor=cA,edgecolor=cA))
            print('#%06x'%tuple_to_bit24(cA))
    draw()

def show_complete_fastest_pallet():
    '''
    16 bit RGB 565 but don't even write the lower byte
    '''
    figure(figsize=(10,5),facecolor=(1,)*4)
    ax=subplot(111)
    subplots_adjust(0,0,1,1,0,0)
    colors = enumerate_fast_colors()
    i = 0;
    xlim(0,8)
    ylim(0,4)
    for k in range(8):
        for ij,j in enumerate([0,2,1,3]):
            i = k*4+j
            c  = tuple_to_bit16(colors[i])
            c0 = bit16_RGB_to_tuple(c & 0b1111111100000000)
            c2 = bit16_RGB_to_tuple((c|0b0000100000001000)&0b1111111100000000)
            cA = .5*(np.array(c0)+np.array(c2))
            ax.add_patch(patches.Rectangle((k,ij),1,1,facecolor=cA,edgecolor=cA))
            print("'#%06x',"%tuple_to_bit24(cA),)
    draw()

def show_hex_pallet(colors):
    figure(figsize=(5,5),facecolor=(1,)*4)
    ax=subplot(111)
    subplots_adjust(0,0,1,1,0,0)
    N = int(ceil(sqrt(len(colors))))
    xlim(0,N)
    ylim(0,N)
    for i,c in enumerate(colors):
        x = i % N
        y = i // N
        ax.add_patch(patches.Rectangle((x,y),1,1,facecolor=c,edgecolor=c))

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
