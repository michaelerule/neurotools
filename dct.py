'''
Discrete cosine transform methods. 

Primarily used with fftzeros code for finding critical points in phase
gradient maps.

'''

from os.path import expanduser
from neurotools.tools import *
from neurotools.parallel import *
from neurotools.signal import *
from neurotools.conv import *
from neurotools.getfftw import *
from matplotlib.mlab import find

"""
def get_grid(data,spacing=0.4):
    '''
    I do not remember what this function did, and it doesn't seem to be
    in use by any other functions, so I'm commenting it out.
    >>> M,N = 10,10
    >>> data = randn(M,N)+1j*randn(M,N)
    >>> gg = getGrid(data)
    >>> dd = abs(gg)
    >>> a  = 4.5/(2*pi)
    >>> kern = sinc(a*dd)*a
    >>> imshow(kern)
    >>> result = ifft2(kern)
    >>> imshow(real(result),interpolation='nearest')
    >>> bone()
    '''
    h,w = shape(data)[:2]
    x1 = list(arange(w+1)*spacing)
    x2 = list(arange(h+1)*spacing)
    x1 = x1+list(reversed(x1[1:-1]))
    x2 = x2+list(reversed(x2[1:-1]))
    x1 = array((x1,)*(h*2))
    x2 = array((x2,)*(w*2)).T
    return x1+1j*x2
"""

def get_mask_antialiased((h,w),aa,spacing,cutoff):
    '''
    Computes a frequency space mask for (h,w) shaped domain with cutoff
    frequency and some anti-aliasing factor as specified by aa. 

    Parameters
    ----------
    h : int
        height
    w : int
        width
    aa : int    
        antialiasing upsampling factor
    spacing : numeric
        array spacing in mm
    cutoff : numeric
        cutoff scale in mm
    
    Retruns:
        h*2 x w*2 symmetric mask to be used with DCT for smoothing
    '''
    assert(aa%2==1)
    f1     = fftfreq(h*2*aa,spacing)
    f2     = fftfreq(w*2*aa,spacing)
    ff     = abs(outer_complex(f1,f2))
    radius = 1./cutoff
    up     = int32(ff<=radius)
    up     = roll(up,aa*h+aa/2,0)
    up     = roll(up,aa*w+aa/2,1)
    mask   = array([[mean(up[j*aa:(j+1)*aa,i*aa:(i+1)*aa]) for i in range(w*2)] for j in range(h*2)])
    mask   = fftshift(mask)
    return mask
    
def get_mask((h,w),spacing,cutoff):
    '''
    Args:
        height: height of array
        width: width of array
        spacing: array spacing in mm
        cutoff: scale in mm
    return: h*2 x w*2 symmetric mask to be used with DCT for smoothing
    '''
    f1     = fftfreq(h*2,spacing)
    f2     = fftfreq(w*2,spacing)
    ff     = abs(outerComplex(f1,f2))
    radius = 1./cutoff
    up     = int32(ff<=radius)
    return up

def dct_cut(data,cutoff,spacing=0.4):
    '''
    Low-pass filters image data by discarding high frequency Fourier 
    components. 
    Image data is reflected before being processes (i.e.) mirrored boundary
    conditions. 
    TODO: I think there is some way to compute the mirrored conditions
    directly without copying the image data.
    
    This function has been superseded by dct_cut_antialias, which is more 
    accurate.
    '''
    print 'WARNING DEPRICATED USE dct_cut_antialias'
    h,w    = shape(data)[:2]
    mask   = getMask((h,w),spacing,cutoff)
    mirror = reflect2D(data)
    ff2    = fft2(mirror,axes=(0,1))
    cut    = (ff2.T*mask.T).T # weird shape broadcasting constraints
    result = ifft2(cut,axes=(0,1))[:h,:w,...]
    return result

def outer_complex(a,b):
    '''
    Not quite outer product, intead of a_i * b_k its 
    a_i + 1j * b_k
    which I guess is like
    log(outer(exp(a_i),exp(1j*b)))
    '''
    return a[...,None]+1j*b[None,...]

def dct_cut_antialias(data,cutoff,spacing=0.4):
    '''
    Uses brute-force upsampling to anti-alias the frequency space sinc
    function, skirting numerical issues that derives either from 
    attempting to evaulate the radial sinc function on a small 2D domain,
    or select a constant frequency cutoff in the frequency space. 
    '''
    '''
    # Sinc experiment
    
    M,N    = 60,80
    data   = randn(M,N)+1j*randn(M,N)
    h,w    = shape(data)[:2]
    
    aa     = 7
    assert(aa%2==1)
    f1     = fftfreq(h*2*aa,spacing)
    f2     = fftfreq(w*2*aa,spacing)
    ff     = abs(outerComplex(f1,f2))
    radius = spacing/cutoff
    up     = int32(ff<=radius)
    #up    = fftshift(up)
    up     = roll(up,aa*h+aa/2,0)
    up     = roll(up,aa*w+aa/2,1)
    mask   = array([[mean(up[j*aa:(j+1)*aa,i*aa:(i+1)*aa]) for i in range(w*2)] for j in range(h*2)])
    mask   = fftshift(mask)
    
    f1B    = fftfreq(h*2,spacing)
    f2B    = fftfreq(w*2,spacing)
    ffB    = abs(outerComplex(f1B,f2B))
    maskB  = int32(ffB<=radius)
    
    mirror = reflect2D(data)
    ff2    = fft2(mirror,axes=(0,1))
    cut    = (ff2.T*mask.T).T # weird shape broadcasting constraints
    result = ifft2(cut,axes=(0,1))[:h,:w,...]
    
    subplot(231)
    imshow(real(up),interpolation='nearest')
    subplot(232)
    kern   = fft2(mask,axes=(0,1))
    imshow(fftshift(real(kern)),interpolation='nearest')
    subplot(233)
    imshow(fftshift(mask),interpolation='nearest')
    subplot(234)
    imshow(real(result),interpolation='nearest')
    subplot(235)
    imshow(fftshift(real(maskB)),interpolation='nearest')
    '''
    h,w    = shape(data)[:2]
    mask   = get_mask_antialiased((h,w),7,spacing,cutoff)
    mirror = reflect2D(data)
    ff2    = fft2(mirror,axes=(0,1))
    cut    = (ff2.T*mask.T).T # weird shape broadcasting constraints
    result = ifft2(cut,axes=(0,1))[:h,:w,...]
    return result

def dct_cut_downsampled(data,cutoff,spacing=0.4):
    '''
    like dctCut but also lowers the sampling rate, creating a compact 
    representation from which the whole downsampled data could be 
    recovered
    '''
    h,w    = shape(data)[:2]
    f1     = fftfreq(h*2,spacing)
    f2     = fftfreq(w*2,spacing)
    wl     = 1./abs(reshape(f1,(2*h,1))+1j*reshape(f2,(1,2*w)))
    mask   = int32(wl>=cutoff)
    mirror = reflect2D(data)
    ff     = fft2(mirror,axes=(0,1))
    cut    = (ff.T*mask.T).T # weird shape broadcasting constraints
    empty_cols = find(all(mask==0,0))
    empty_rows = find(all(mask==0,1))
    delete_col = len(empty_cols)/2 #idiv important here
    delete_row = len(empty_rows)/2 #idiv important here
    keep_cols  = w-delete_col
    keep_rows  = h-delete_row
    col_mask = zeros(w*2)
    col_mask[:keep_cols] =1
    col_mask[-keep_cols:]=1
    col_mask = col_mask==1
    row_mask = zeros(h*2)
    row_mask[:keep_rows] =1
    row_mask[-keep_rows:]=1
    row_mask = row_mask==1
    cut = cut[row_mask,...][:,col_mask,...]
    w,h = keep_cols,keep_rows
    result = ifft2(cut,axes=(0,1))[:h,:w,...]
    return result

def dct_upsample(data,factor=2):
    '''
    Uses the DCT to upsample array data. Nice for visualization. Uses a
    discrete cosine transform to smoothly upsample image data. Boundary
    conditions are handeled as reflected. 
    
    Data is made symmetric, fourier transformed, then inserted into the 
    low-frequency components of a larger array, which is then inverse 
    transformed. 
    
    This could probably be optimized with a customized array implimentation
    to avoid the copying.
    
    Parameters:
        data (ndarray): frist two dimensions should be height and width. 
            data may contain aribtrary number of additional dimensions
        factor (int): upsampling factor.
    
    Test code    
    grid = arange(10)&1
    grid = grid[None,:]^grid[:,None]
    amp  = dct_upsample(grid,UPSAMPLE).real 
    '''
    h,w = shape(data)[:2]
    mirrored = reflect2D_1(data)
    ff = fft2(mirrored,axes=(0,1))
    h2,w2 = shape(mirrored)[:2]
    newshape = (h2*factor,w2*factor) + shape(data)[2:]
    result = zeros(newshape,dtype=ff.dtype)
    H = h+1#have to add one more on the left to grab the nyqist term
    W = w+1
    result[:H ,:W ,...]=ff[:H , :W,...]
    result[-h:,:W ,...]=ff[-h:, :W,...]
    result[:H ,-w:,...]=ff[:H ,-w:,...]
    result[-h:,-w:,...]=ff[-h:,-w:,...]
    result = ifft2(result,axes=(0,1))
    result = result[:h*factor-factor+1,:w*factor-factor+1,...]
    return result*(factor*factor)

def iterated_upsample(data,niter=1):
    for i in range(niter):
        data = dct_upsample(data)
    return data



def dct_upsample_notrim(data,factor=2):
    '''
    Uses the DCT to upsample array data. Nice for visualization. Uses a
    discrete cosine transform to smoothly upsample image data. Boundary
    conditions are handeled as reflected. 
    
    Data is made symmetric, fourier transformed, then inserted into the 
    low-frequency components of a larger array, which is then inverse 
    transformed. 
    
    This could probably be optimized with a customized array implimentation
    to avoid the copying.
    
    Parameters:
        data (ndarray): frist two dimensions should be height and width. 
            data may contain aribtrary number of additional dimensions
        factor (int): upsampling factor.
    '''
    print 'ALIGNMENT BUG DO NOT USE YET'
    assert 0
    h,w = shape(data)[:2]
    mirrored = reflect2D_1(data)
    ff = fft2(mirrored,axes=(0,1))
    h2,w2 = shape(mirrored)[:2]
    newshape = (h2*factor,w2*factor) + shape(data)[2:]
    result = zeros(newshape,dtype=ff.dtype)
    H = h+1#have to add one more on the left to grab the nyqist term
    W = w+1
    result[:H ,:W ,...]=ff[:H , :W,...]
    result[-h:,:W ,...]=ff[-h:, :W,...]
    result[:H ,-w:,...]=ff[:H ,-w:,...]
    result[-h:,-w:,...]=ff[-h:,-w:,...]
    result = ifft2(result,axes=(0,1))
    result = result[0:h*factor+factor,0:w*factor+factor,...]
    return result*(factor*factor)






