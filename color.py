#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
Miscellaneous color-related functions. Several color maps for use with
matplotlib. A couple idiosyncratic color pallets.

Defines the color maps parula, isolum, and extended

This class also defines three hue wheel color maps of varying brightness

>>> lighthue = mcolors.ListedColormap(lighthues(NCMAP),'lighthue')
>>> medhue   = mcolors.ListedColormap(medhues  (NCMAP),'medhue')
>>> darkhue  = mcolors.ListedColormap(darkhues (NCMAP),'darkhue')
'''

import pylab
from neurotools.tools  import *
from neurotools.signal import *
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os.path    import expanduser
from matplotlib import cm
from matplotlib import *
from pylab import *

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
INDEGO     = [0.37843053,  0.4296282 ,  0.76422011]
VERIDIAN   = [0.06695279,  0.74361409,  0.55425139]
CHARTREUSE = [0.71152929,  0.62526339,  0.10289384]
CRIMSON    = [0.84309675,  0.37806273,  0.32147779]



#############################################################################
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
    return dot(luminance_matrix(method=method),(r,g,b))

def luminance_matrix(method='perceived'):
    '''
    Method 'perceived' 0.299*R + 0.587*G + 0.114*B
    Method 'standard'  0.2126*R + 0.7152*G + 0.0722*B
    Method 'lightness' 0.3*R + 0.59*G + 0.11*B
    '''
    if method=='standard':
        x1 = 0.2126
        x2 = 0.7152
        x3 = 0.0722
    elif method=='perceived':
        x1 = 0.299
        x2 = 0.587
        x3 = 0.114
    elif method=='lightness':
        x1 = 0.30
        x2 = 0.59
        x3 = 0.11
    else:
        assert 0
    LRGB = np.array([x1,x2,x3])
    LRGB = LRGB / np.sum(LRGB)
    return LRGB

def match_luminance(target,color,THRESHOLD=0.01,squared=False,method='perceived'):
    '''
    Adjust color to match luminosity of target

    Method 'perceived' 0.299*R + 0.587*G + 0.114*B
    Method 'standard'  0.2126*R + 0.7152*G + 0.0722*B
    Method 'lightness' 0.3*R + 0.59*G + 0.11*B
    '''
    LRGB   = luminance_matrix(method)
    color  = array(color)
    target = array(target)
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
        correction = 0.5*(arithmetic_corrected+geometric_corrected)
        color = 0.9*color+0.1*np.clip(correction,0,1)
        if squared:
            source    = np.dot(LRGB,color**2)**0.5
        else:
            source    = np.dot(LRGB,color)
    return color

def rotate(colors,th):
    '''
    Rotate a list of rgb colors by angle theta
    '''
    Q1 = sin(th)/sqrt(3)
    Q2 = (1-cos(th))/3
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
    alpha  = 0.5*(2*r-g-b)
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

def hcl2rgb(h,c,l,target = 1.0):
    '''
    h: hue
    c: chroma
    l: luminosity
    '''
    LRGB = luminance_matrix()
    h = h*pi/180.0
    alpha = cos(h)
    beta = sin(h)*2/sqrt(3)
    B = l - x1*(alpha+beta/2)-x2*beta
    R = alpha + beta/2+B
    G = beta+B
    RGB = np.array([R,G,B])
    '''
    RGB = RGB/np.max(RGB)
    # old code forced luminance conversion. No longer doin this
    luminance = np.dot(LRGB,RGB)
    # luminance will be off target. why?
    if luminance<target:
        # the color is not as bright as it needs to be.
        # blend in white
        lw = np.dot(LRGB,ones(3))
        # solve convex combination:
        # alpha*luminance+(1-alpha)*lw = target
        # a*l+lw-a*lw = t
        # a*(l-lw)+lw = t
        # a*(l-lw) = t-lw
        # a = (t-lw)/(l-lw)
        a = (target-lw)/(luminance-lw)
        RGB = a*RGB + (1-a)*ones(3)
    elif luminance>target:
        # the color is too bright, blend with black
        BLACK = zeros(3)
        lb = np.dot(LRGB,BLACK)
        # solve the convex combination
        # a*l+(1-a)*lb = t
        # a = (t-lb)/(l-lb)
        a = (target-lb)/(luminance-lb)
        RGB = a*RGB + (1-a)*BLACK
    '''
    return clip(RGB,0,1)

def circularly_smooth_colormap(cm,s):
    '''
    Smooth a colormap with cirular boundary conditions

    s: sigma, standard deviation of gaussian smoothing kernel in samples
    cm: color map, array-like of RGB tuples
    '''
    # Do circular boundary conditions the lazy way
    cm = array(cm)
    N = shape(cm)[0]
    cm = concatenate([cm,cm,cm])
    R,G,B = cm.T
    R = gaussian_smooth(R,s)
    G = gaussian_smooth(G,s)
    B = gaussian_smooth(B,s)
    RGB = array([R,G,B]).T
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
    x1 = 0.30
    x2 = 0.59
    x3 = 0.11
    LRGB  = array([x1,x2,x3])
    alpha = cos(h)
    beta  = sin(h)*2/sqrt(3)
    B = 1.0 - x1*(alpha+beta/2)-x2*beta
    R = alpha + beta/2+B
    G = beta+B
    RGB = array([R,G,B])
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
    h = ravel(int32(angle(z)*N/(2*pi))%N)
    v = ravel(int32(clip(abs(z),0,1)*(N-1)))
    return reshape(__HL_LUT__[h,v],shape(z)+(3,))




#############################################################################
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


#############################################################################
# Three isoluminance hue wheels at varying brightness
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

#############################################################################
# Isoluminance hue wheel color map
# pip install husl
# http://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/28790/versions/5/screenshot.jpg
# https://pypi.python.org/pypi/husl
# https://mycarta.wordpress.com/2014/10/30/new-matlab-isoluminant-colormap-for-azimuth-data/
# radius = 38; % chroma
# theta = linspace(0, 2*pi, 256)'; % hue
# a = radius * cos(theta);
# b = radius * sin(theta);
# L = (ones(1, 256)*65)'; % lightness
# Lab = [L, a, b];
# RGB=colorspace('RGB<-Lab',Lab(end:-1:1,:));
# https://mycarta.wordpress.com/
isolum_data = [
[0.8658,0.5133,0.6237],[0.8638,0.5137,0.6301],[0.8615,0.5141,0.6366],
[0.8591,0.5147,0.6430],[0.8566,0.5153,0.6494],[0.8538,0.5160,0.6557],
[0.8509,0.5168,0.6621],[0.8479,0.5176,0.6684],[0.8446,0.5185,0.6747],
[0.8412,0.5195,0.6810],[0.8376,0.5206,0.6872],[0.8338,0.5218,0.6934],
[0.8299,0.5230,0.6996],[0.8258,0.5243,0.7057],[0.8215,0.5257,0.7118],
[0.8171,0.5271,0.7178],[0.8125,0.5286,0.7237],[0.8077,0.5302,0.7296],
[0.8028,0.5318,0.7355],[0.7977,0.5335,0.7412],[0.7924,0.5352,0.7469],
[0.7870,0.5370,0.7525],[0.7814,0.5389,0.7581],[0.7756,0.5408,0.7635],
[0.7697,0.5427,0.7689],[0.7636,0.5447,0.7742],[0.7573,0.5467,0.7794],
[0.7509,0.5488,0.7845],[0.7443,0.5509,0.7895],[0.7376,0.5530,0.7944],
[0.7307,0.5552,0.7992],[0.7237,0.5574,0.8039],[0.7165,0.5596,0.8085],
[0.7092,0.5618,0.8130],[0.7017,0.5641,0.8173],[0.6940,0.5664,0.8216],
[0.6862,0.5687,0.8257],[0.6783,0.5710,0.8297],[0.6702,0.5733,0.8336],
[0.6620,0.5757,0.8373],[0.6536,0.5780,0.8409],[0.6451,0.5804,0.8444],
[0.6365,0.5828,0.8477],[0.6277,0.5851,0.8509],[0.6188,0.5875,0.8540],
[0.6097,0.5898,0.8569],[0.6005,0.5922,0.8597],[0.5912,0.5946,0.8623],
[0.5818,0.5969,0.8648],[0.5722,0.5992,0.8672],[0.5625,0.6016,0.8693],
[0.5527,0.6039,0.8714],[0.5427,0.6062,0.8733],[0.5326,0.6085,0.8750],
[0.5224,0.6107,0.8766],[0.5121,0.6130,0.8780],[0.5017,0.6152,0.8792],
[0.4911,0.6174,0.8803],[0.4804,0.6196,0.8813],[0.4696,0.6218,0.8820],
[0.4587,0.6239,0.8826],[0.4476,0.6260,0.8831],[0.4365,0.6281,0.8834],
[0.4252,0.6302,0.8835],[0.4138,0.6322,0.8835],[0.4022,0.6342,0.8833],
[0.3905,0.6362,0.8829],[0.3787,0.6381,0.8824],[0.3668,0.6400,0.8817],
[0.3547,0.6419,0.8809],[0.3425,0.6437,0.8799],[0.3301,0.6455,0.8787],
[0.3175,0.6473,0.8774],[0.3048,0.6491,0.8759],[0.2918,0.6508,0.8742],
[0.2787,0.6524,0.8724],[0.2653,0.6541,0.8705],[0.2517,0.6557,0.8684],
[0.2377,0.6572,0.8661],[0.2235,0.6588,0.8637],[0.2088,0.6602,0.8611],
[0.1937,0.6617,0.8584],[0.1781,0.6631,0.8556],[0.1617,0.6645,0.8526],
[0.1444,0.6658,0.8494],[0.1260,0.6671,0.8461],[0.1059,0.6684,0.8427],
[0.0832,0.6696,0.8391],[0.0560,0.6708,0.8354],[0.0206,0.6720,0.8316],
[0.0000,0.6739,0.8282],[0.0000,0.6767,0.8255],[0.0000,0.6794,0.8226],
[0.0000,0.6819,0.8194],[0.0000,0.6841,0.8161],[0.0000,0.6862,0.8124],
[0.0000,0.6880,0.8086],[0.0000,0.6897,0.8045],[0.0000,0.6912,0.8003],
[0.0000,0.6925,0.7958],[0.0000,0.6936,0.7911],[0.0000,0.6946,0.7862],
[0.0000,0.6953,0.7810],[0.0000,0.6959,0.7757],[0.0000,0.6964,0.7702],
[0.0000,0.6966,0.7645],[0.0000,0.6968,0.7586],[0.0000,0.6967,0.7525],
[0.0000,0.6965,0.7461],[0.0000,0.6962,0.7396],[0.0000,0.6957,0.7330],
[0.0000,0.6950,0.7261],[0.0000,0.6942,0.7190],[0.0000,0.6933,0.7118],
[0.0000,0.6922,0.7043],[0.0000,0.6909,0.6967],[0.0000,0.6896,0.6889],
[0.0221,0.6893,0.6821],[0.0525,0.6894,0.6757],[0.0765,0.6895,0.6693],
[0.0965,0.6895,0.6629],[0.1140,0.6895,0.6564],[0.1300,0.6895,0.6499],
[0.1448,0.6894,0.6434],[0.1588,0.6893,0.6369],[0.1720,0.6892,0.6304],
[0.1847,0.6890,0.6238],[0.1968,0.6889,0.6173],[0.2086,0.6886,0.6107],
[0.2200,0.6884,0.6042],[0.2311,0.6881,0.5976],[0.2419,0.6878,0.5911],
[0.2524,0.6874,0.5846],[0.2627,0.6870,0.5780],[0.2728,0.6866,0.5716],
[0.2827,0.6862,0.5651],[0.2925,0.6857,0.5587],[0.3020,0.6852,0.5523],
[0.3115,0.6846,0.5459],[0.3207,0.6841,0.5396],[0.3299,0.6835,0.5333],
[0.3389,0.6828,0.5271],[0.3478,0.6821,0.5209],[0.3566,0.6814,0.5147],
[0.3653,0.6807,0.5087],[0.3739,0.6799,0.5026],[0.3824,0.6791,0.4967],
[0.3908,0.6783,0.4908],[0.3992,0.6774,0.4850],[0.4074,0.6765,0.4793],
[0.4156,0.6756,0.4736],[0.4237,0.6746,0.4680],[0.4317,0.6736,0.4625],
[0.4397,0.6726,0.4571],[0.4476,0.6716,0.4518],[0.4554,0.6705,0.4466],
[0.4632,0.6693,0.4415],[0.4709,0.6682,0.4365],[0.4785,0.6670,0.4316],
[0.4861,0.6658,0.4268],[0.4937,0.6645,0.4221],[0.5012,0.6633,0.4176],
[0.5086,0.6619,0.4131],[0.5160,0.6606,0.4088],[0.5234,0.6592,0.4046],
[0.5306,0.6578,0.4005],[0.5379,0.6564,0.3966],[0.5451,0.6549,0.3928],
[0.5522,0.6534,0.3891],[0.5593,0.6519,0.3856],[0.5664,0.6503,0.3822],
[0.5734,0.6487,0.3790],[0.5803,0.6471,0.3759],[0.5872,0.6454,0.3730],
[0.5941,0.6437,0.3702],[0.6009,0.6420,0.3676],[0.6077,0.6403,0.3652],
[0.6144,0.6385,0.3629],[0.6211,0.6367,0.3607],[0.6277,0.6349,0.3588],
[0.6342,0.6331,0.3570],[0.6407,0.6312,0.3553],[0.6472,0.6293,0.3539],
[0.6536,0.6273,0.3526],[0.6600,0.6254,0.3515],[0.6662,0.6234,0.3505],
[0.6725,0.6214,0.3498],[0.6787,0.6194,0.3492],[0.6848,0.6174,0.3488],
[0.6908,0.6153,0.3485],[0.6968,0.6132,0.3485],[0.7027,0.6111,0.3486],
[0.7086,0.6090,0.3489],[0.7144,0.6069,0.3493],[0.7201,0.6047,0.3500],
[0.7258,0.6026,0.3508],[0.7314,0.6004,0.3518],[0.7369,0.5982,0.3529],
[0.7423,0.5960,0.3542],[0.7477,0.5938,0.3557],[0.7529,0.5916,0.3574],
[0.7581,0.5893,0.3592],[0.7633,0.5871,0.3612],[0.7683,0.5849,0.3633],
[0.7732,0.5827,0.3656],[0.7781,0.5804,0.3680],[0.7828,0.5782,0.3706],
[0.7875,0.5760,0.3734],[0.7921,0.5737,0.3763],[0.7966,0.5715,0.3793],
[0.8010,0.5693,0.3825],[0.8052,0.5671,0.3858],[0.8094,0.5649,0.3892],
[0.8135,0.5628,0.3928],[0.8175,0.5606,0.3965],[0.8213,0.5585,0.4004],
[0.8251,0.5564,0.4043],[0.8287,0.5543,0.4084],[0.8322,0.5522,0.4126],
[0.8356,0.5502,0.4169],[0.8389,0.5482,0.4214],[0.8421,0.5462,0.4259],
[0.8451,0.5443,0.4306],[0.8481,0.5424,0.4353],[0.8509,0.5405,0.4402],
[0.8535,0.5387,0.4451],[0.8561,0.5369,0.4502],[0.8585,0.5352,0.4553],
[0.8607,0.5335,0.4605],[0.8629,0.5318,0.4658],[0.8649,0.5303,0.4712],
[0.8667,0.5287,0.4767],[0.8685,0.5273,0.4822],[0.8700,0.5259,0.4879],
[0.8715,0.5245,0.4936],[0.8728,0.5232,0.4993],[0.8739,0.5220,0.5051],
[0.8749,0.5209,0.5110],[0.8757,0.5198,0.5170],[0.8764,0.5188,0.5230],
[0.8770,0.5178,0.5290],[0.8773,0.5170,0.5351],[0.8776,0.5162,0.5413],
[0.8776,0.5155,0.5475],[0.8775,0.5148,0.5537],[0.8773,0.5143,0.5599],
[0.8769,0.5138,0.5662],[0.8763,0.5134,0.5725],[0.8756,0.5131,0.5789],
[0.8747,0.5129,0.5853],[0.8736,0.5128,0.5916],[0.8724,0.5127,0.5980],
[0.8710,0.5127,0.6045],[0.8695,0.5128,0.6109],[0.8677,0.5130,0.6173],
[0.8658,0.5133,0.6237]]

try:
    isolum = matplotlib.colors.ListedColormap(isolum_data,'isolum')
    plt.register_cmap(name='isolum', cmap=isolum)
    isolum_data = np.float32(isolum_data)

    double_isolum_data = concatenate([isolum_data[::2],isolum_data[::2]])
    double_isolum = matplotlib.colors.ListedColormap(double_isolum_data,'isolum')
    plt.register_cmap(name='double_isolum', cmap=double_isolum)

except:
    pass
    #sphinx workaround



#############################################################################
# Parula color map
'''
Provides the Matlab(r)(tm) parula color map for matplotlib.
Mathworks claims copyright on the parula color scheme.
Accordingly, you may only use this file and this color map if you currently
own a lisence for a Matlab version that contains the Parula color map.
Please see hsvtools.py for non-copyrighted colormaps of constant luminance
and constant luminance gradient.
Some have argued that since Parula arises from constraint optimization
for constant luminance and color-blindness, it may not be subject to
copyright. If you can show analytically that the Parula color map is the
unique solution to an optimization problem, it is likely that the copyright
can be safely ignored. (However, check patents! it could be patented)
However, I have not done this math so I am riterating that, as far as
currently known, Mathwork's copyright claim on the parula color map stands.
'''
import matplotlib
from matplotlib.cm import *
from numpy import *
parula_data = [
[0.2081, 0.1663, 0.5292],[0.2091, 0.1721, 0.5411],[0.2101, 0.1779, 0.5530],
[0.2109, 0.1837, 0.5650],[0.2116, 0.1895, 0.5771],[0.2121, 0.1954, 0.5892],
[0.2124, 0.2013, 0.6013],[0.2125, 0.2072, 0.6135],[0.2123, 0.2132, 0.6258],
[0.2118, 0.2192, 0.6381],[0.2111, 0.2253, 0.6505],[0.2099, 0.2315, 0.6629],
[0.2084, 0.2377, 0.6753],[0.2063, 0.2440, 0.6878],[0.2038, 0.2503, 0.7003],
[0.2006, 0.2568, 0.7129],[0.1968, 0.2632, 0.7255],[0.1921, 0.2698, 0.7381],
[0.1867, 0.2764, 0.7507],[0.1802, 0.2832, 0.7634],[0.1728, 0.2902, 0.7762],
[0.1641, 0.2975, 0.7890],[0.1541, 0.3052, 0.8017],[0.1427, 0.3132, 0.8145],
[0.1295, 0.3217, 0.8269],[0.1147, 0.3306, 0.8387],[0.0986, 0.3397, 0.8495],
[0.0816, 0.3486, 0.8588],[0.0646, 0.3572, 0.8664],[0.0482, 0.3651, 0.8722],
[0.0329, 0.3724, 0.8765],[0.0213, 0.3792, 0.8796],[0.0136, 0.3853, 0.8815],
[0.0086, 0.3911, 0.8827],[0.0060, 0.3965, 0.8833],[0.0051, 0.4017, 0.8834],
[0.0054, 0.4066, 0.8831],[0.0067, 0.4113, 0.8825],[0.0089, 0.4159, 0.8816],
[0.0116, 0.4203, 0.8805],[0.0148, 0.4246, 0.8793],[0.0184, 0.4288, 0.8779],
[0.0223, 0.4329, 0.8763],[0.0264, 0.4370, 0.8747],[0.0306, 0.4410, 0.8729],
[0.0349, 0.4449, 0.8711],[0.0394, 0.4488, 0.8692],[0.0437, 0.4526, 0.8672],
[0.0477, 0.4564, 0.8652],[0.0514, 0.4602, 0.8632],[0.0549, 0.4640, 0.8611],
[0.0582, 0.4677, 0.8589],[0.0612, 0.4714, 0.8568],[0.0640, 0.4751, 0.8546],
[0.0666, 0.4788, 0.8525],[0.0689, 0.4825, 0.8503],[0.0710, 0.4862, 0.8481],
[0.0729, 0.4899, 0.8460],[0.0746, 0.4937, 0.8439],[0.0761, 0.4974, 0.8418],
[0.0773, 0.5012, 0.8398],[0.0782, 0.5051, 0.8378],[0.0789, 0.5089, 0.8359],
[0.0794, 0.5129, 0.8341],[0.0795, 0.5169, 0.8324],[0.0793, 0.5210, 0.8308],
[0.0788, 0.5251, 0.8293],[0.0778, 0.5295, 0.8280],[0.0764, 0.5339, 0.8270],
[0.0746, 0.5384, 0.8261],[0.0724, 0.5431, 0.8253],[0.0698, 0.5479, 0.8247],
[0.0668, 0.5527, 0.8243],[0.0636, 0.5577, 0.8239],[0.0600, 0.5627, 0.8237],
[0.0562, 0.5677, 0.8234],[0.0523, 0.5727, 0.8231],[0.0484, 0.5777, 0.8228],
[0.0445, 0.5826, 0.8223],[0.0408, 0.5874, 0.8217],[0.0372, 0.5922, 0.8209],
[0.0342, 0.5968, 0.8198],[0.0317, 0.6012, 0.8186],[0.0296, 0.6055, 0.8171],
[0.0279, 0.6097, 0.8154],[0.0265, 0.6137, 0.8135],[0.0255, 0.6176, 0.8114],
[0.0248, 0.6214, 0.8091],[0.0243, 0.6250, 0.8066],[0.0239, 0.6285, 0.8039],
[0.0237, 0.6319, 0.8010],[0.0235, 0.6352, 0.7980],[0.0233, 0.6384, 0.7948],
[0.0231, 0.6415, 0.7916],[0.0230, 0.6445, 0.7881],[0.0229, 0.6474, 0.7846],
[0.0227, 0.6503, 0.7810],[0.0227, 0.6531, 0.7773],[0.0232, 0.6558, 0.7735],
[0.0238, 0.6585, 0.7696],[0.0246, 0.6611, 0.7656],[0.0263, 0.6637, 0.7615],
[0.0282, 0.6663, 0.7574],[0.0306, 0.6688, 0.7532],[0.0338, 0.6712, 0.7490],
[0.0373, 0.6737, 0.7446],[0.0418, 0.6761, 0.7402],[0.0467, 0.6784, 0.7358],
[0.0516, 0.6808, 0.7313],[0.0574, 0.6831, 0.7267],[0.0629, 0.6854, 0.7221],
[0.0692, 0.6877, 0.7173],[0.0755, 0.6899, 0.7126],[0.0820, 0.6921, 0.7078],
[0.0889, 0.6943, 0.7029],[0.0956, 0.6965, 0.6979],[0.1031, 0.6986, 0.6929],
[0.1104, 0.7007, 0.6878],[0.1180, 0.7028, 0.6827],[0.1258, 0.7049, 0.6775],
[0.1335, 0.7069, 0.6723],[0.1418, 0.7089, 0.6669],[0.1499, 0.7109, 0.6616],
[0.1585, 0.7129, 0.6561],[0.1671, 0.7148, 0.6507],[0.1758, 0.7168, 0.6451],
[0.1849, 0.7186, 0.6395],[0.1938, 0.7205, 0.6338],[0.2033, 0.7223, 0.6281],
[0.2128, 0.7241, 0.6223],[0.2224, 0.7259, 0.6165],[0.2324, 0.7275, 0.6107],
[0.2423, 0.7292, 0.6048],[0.2527, 0.7308, 0.5988],[0.2631, 0.7324, 0.5929],
[0.2735, 0.7339, 0.5869],[0.2845, 0.7354, 0.5809],[0.2953, 0.7368, 0.5749],
[0.3064, 0.7381, 0.5689],[0.3177, 0.7394, 0.5630],[0.3289, 0.7406, 0.5570],
[0.3405, 0.7417, 0.5512],[0.3520, 0.7428, 0.5453],[0.3635, 0.7438, 0.5396],
[0.3753, 0.7446, 0.5339],[0.3869, 0.7454, 0.5283],[0.3986, 0.7461, 0.5229],
[0.4103, 0.7467, 0.5175],[0.4218, 0.7473, 0.5123],[0.4334, 0.7477, 0.5072],
[0.4447, 0.7482, 0.5021],[0.4561, 0.7485, 0.4972],[0.4672, 0.7487, 0.4924],
[0.4783, 0.7489, 0.4877],[0.4892, 0.7491, 0.4831],[0.5000, 0.7491, 0.4786],
[0.5106, 0.7492, 0.4741],[0.5212, 0.7492, 0.4698],[0.5315, 0.7491, 0.4655],
[0.5418, 0.7490, 0.4613],[0.5519, 0.7489, 0.4571],[0.5619, 0.7487, 0.4531],
[0.5718, 0.7485, 0.4490],[0.5816, 0.7482, 0.4451],[0.5913, 0.7479, 0.4412],
[0.6009, 0.7476, 0.4374],[0.6103, 0.7473, 0.4335],[0.6197, 0.7469, 0.4298],
[0.6290, 0.7465, 0.4261],[0.6382, 0.7460, 0.4224],[0.6473, 0.7456, 0.4188],
[0.6564, 0.7451, 0.4152],[0.6653, 0.7446, 0.4116],[0.6742, 0.7441, 0.4081],
[0.6830, 0.7435, 0.4046],[0.6918, 0.7430, 0.4011],[0.7004, 0.7424, 0.3976],
[0.7091, 0.7418, 0.3942],[0.7176, 0.7412, 0.3908],[0.7261, 0.7405, 0.3874],
[0.7346, 0.7399, 0.3840],[0.7430, 0.7392, 0.3806],[0.7513, 0.7385, 0.3773],
[0.7596, 0.7378, 0.3739],[0.7679, 0.7372, 0.3706],[0.7761, 0.7364, 0.3673],
[0.7843, 0.7357, 0.3639],[0.7924, 0.7350, 0.3606],[0.8005, 0.7343, 0.3573],
[0.8085, 0.7336, 0.3539],[0.8166, 0.7329, 0.3506],[0.8246, 0.7322, 0.3472],
[0.8325, 0.7315, 0.3438],[0.8405, 0.7308, 0.3404],[0.8484, 0.7301, 0.3370],
[0.8563, 0.7294, 0.3336],[0.8642, 0.7288, 0.3300],[0.8720, 0.7282, 0.3265],
[0.8798, 0.7276, 0.3229],[0.8877, 0.7271, 0.3193],[0.8954, 0.7266, 0.3156],
[0.9032, 0.7262, 0.3117],[0.9110, 0.7259, 0.3078],[0.9187, 0.7256, 0.3038],
[0.9264, 0.7256, 0.2996],[0.9341, 0.7256, 0.2953],[0.9417, 0.7259, 0.2907],
[0.9493, 0.7264, 0.2859],[0.9567, 0.7273, 0.2808],[0.9639, 0.7285, 0.2754],
[0.9708, 0.7303, 0.2696],[0.9773, 0.7326, 0.2634],[0.9831, 0.7355, 0.2570],
[0.9882, 0.7390, 0.2504],[0.9922, 0.7431, 0.2437],[0.9952, 0.7476, 0.2373],
[0.9973, 0.7524, 0.2310],[0.9986, 0.7573, 0.2251],[0.9991, 0.7624, 0.2195],
[0.9990, 0.7675, 0.2141],[0.9985, 0.7726, 0.2090],[0.9976, 0.7778, 0.2042],
[0.9964, 0.7829, 0.1995],[0.9950, 0.7880, 0.1949],[0.9933, 0.7931, 0.1905],
[0.9914, 0.7981, 0.1863],[0.9894, 0.8032, 0.1821],[0.9873, 0.8083, 0.1780],
[0.9851, 0.8133, 0.1740],[0.9828, 0.8184, 0.1700],[0.9805, 0.8235, 0.1661],
[0.9782, 0.8286, 0.1622],[0.9759, 0.8337, 0.1583],[0.9736, 0.8389, 0.1544],
[0.9713, 0.8441, 0.1505],[0.9692, 0.8494, 0.1465],[0.9672, 0.8548, 0.1425],
[0.9654, 0.8603, 0.1385],[0.9638, 0.8659, 0.1343],[0.9623, 0.8716, 0.1301],
[0.9611, 0.8774, 0.1258],[0.9600, 0.8834, 0.1215],[0.9593, 0.8895, 0.1171],
[0.9588, 0.8958, 0.1126],[0.9586, 0.9022, 0.1082],[0.9587, 0.9088, 0.1036],
[0.9591, 0.9155, 0.0990],[0.9599, 0.9225, 0.0944],[0.9610, 0.9296, 0.0897],
[0.9624, 0.9368, 0.0850],[0.9641, 0.9443, 0.0802],[0.9662, 0.9518, 0.0753],
[0.9685, 0.9595, 0.0703],[0.9710, 0.9673, 0.0651],[0.9736, 0.9752, 0.0597],
[0.9763, 0.9831, 0.0538]]

try:
    parula = matplotlib.colors.ListedColormap(parula_data,'parula')
    register_cmap(name='parula', cmap=parula)
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
try:
    extended_data = np.float32(extended_data)/255
    extended = matplotlib.colors.ListedColormap(extended_data,'extended')
    register_cmap(name='extended', cmap=extended)

    #############################################################################
    # A balanced colormap that combines aspects of HSV, HSL, and
    # Matteo Niccoli's isoluminant hue wheel, with a little smoothing
    #
    x = np.array([hsv2rgb(h,1,1) for h in np.linspace(0,1,256)*360])
    y = np.array([hcl2rgb(h,1,0.75) for h in np.linspace(0,1,256)*360])
    z = isolum_data[::-1]
    balance_data = circularly_smooth_colormap(x*0.2+y*0.4+z*0.4,30)
    balance = matplotlib.colors.ListedColormap(balance_data,'extended')
    register_cmap(name='balance', cmap=balance)
except:
    extended=None
    isolum=None
    balance=None
    pass
    #sphinx bug workaround

#############################################################################
# Bit-packed color fiddling

def hex_pack_BGR(RGB):
    '''
    Packs RGB colors data in hexadecimal BGR format for fast rendering to
    Javascript canvas.
    RGB: 256x3 RGB array-like, 0..1 values
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
            cA = 0.5*(array(c0)+array(c2))
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
            cA = 0.5*(array(c0)+array(c2))
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
            cA = 0.5*(array(c0)+array(c2))
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
