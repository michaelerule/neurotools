
#execfile(expanduser('~/Dropbox/bin/getfftw.py'))
import getfftw
from scipy.signal import convolve2d
from numpy import *
from functions import npdf

def reflect2D(data):
    '''
    Reflects 2D data ... used in the discrete cosine transform.
    data may have dimensions HxW or HxWxN
    return 2Hx2W or 2Hx2WxN respectively
    '''
    h,w = shape(data)[:2]
    dtype = data.dtype
    if len(shape(data))==2:
        result = zeros((h*2,w*2),dtype=dtype)
    else:
        #assert len(shape(data))==3
        h,w = shape(data)[:2]
        result = zeros((h*2,w*2)+shape(data)[2:],dtype=dtype)
    result[:h,:w,...]=data
    result[h:,:w,...]=flipud(data)
    result[ :,w:,...]=result[:,w-1::-1,...]
    return result


def reflect2D_1(data):
    '''
    Reflects 2D data, without doubling the data on the edge
    data may have dimensions HxW or HxWxN
    return 2H-2x2W-2 or 2H-2x2W-2xN respectively
    '''
    h,w = shape(data)[:2]
    dtype = data.dtype
    if len(shape(data))==2:
        result = zeros((h*2-2,w*2-2),dtype=dtype)
    else:
        h,w = shape(data)[:2]
        result = zeros((h*2-2,w*2-2)+shape(data)[2:],dtype=dtype)
    # top left corner is easy: just a copy of the data
    result[:h,:w,...]=data
    # next do the bottom left. the bottom row gets duplicated unless
    # we trim it off
    result[h:,:w,...]=flipud(data[:-1,:])[:-1,:]
    # then, copy over what we just did. dont copy the last column (which
    # becomes the first column when flipped)
    result[ :,w:,...]=result[:,w-2:0:-1,...]
    return result

def mirror2d(x):
    h,w = shape(x)
    mirrored = zeros((h*2,w*2),dtype=x.dtype)
    mirrored[:h,:w]=x
    mirrored[h:,:w]=flipud(x)
    mirrored[: ,w:]=fliplr(mirrored[:,:w])
    return mirrored

def convolve2dct(x,k):
    h,w = shape(x)
    x = mirror2d(x)
    x = convolve2d(x,k,'same')
    return x[:h,:w]

def FTWMirrorConvolve(x,k):
    '''
    FFT convolve using FTW library. Real data. Mirrored boundaries.
    '''
    pass

def separable2d(X,k,k2=None):
    h,w = shape(X)
    X = mirror2d(X)
    y = array([convolve(x,k,'same') for x in X])
    if k2==None: k2=k
    y = array([convolve(x,k2,'same') for x in y.T]).T
    return y[:h,:w]

def gausskern2d(sigma,size):
    k = size/2
    x = float32(arange(-k,k+1))
    p = npdf(0,sigma,x)
    kern = outer(p,p)
    return kern / sum(kern)

def gausskern1d(sigma,size):
    k = size/2
    x = float32(arange(-k,k+1))
    kern = npdf(0,sigma,x)
    return kern / sum(kern)

def padKern(k,N):
    '''
    NOT IMPLEMENTED
    '''
    l = len(k)
    #middle = 
    pass

def sepConv2dFTW(x,k):
    '''
    NOT IMPLEMENTED USE np.convolve2d or fftconvolve
    reflection padded separable convoltuion
    using FFT
    :param x:
    :param y:
        -- pad kernel to height / width
        -- perform 1D fft in x, multiply (invert?)
        -- perform 1D fft in y, multiple, invert
    '''
    pass


