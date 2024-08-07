#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import numpy as np
from scipy import fft

def blurkernel(L,σ,normalize=False):
    '''
    1D Gaussian blur convolution kernel    
    
    Parameters
    ----------
    L: int
        Size of L×L spatial domain
    σ: positive float
        kernel radius exp(-x²/σ) (standard deviation in x 
        and y ×⎷2)
    normalize: boolean
        Whether to make kernel sum to 1
    '''
    k = np.exp(-(np.arange(-L//2,L//2)/σ)**2)
    if normalize: 
        k /= sum(k)
    return fft.fftshift(k)

def blurkernel2D(L,σ,normalize=False):
    '''
    2D Gaussian blur convolution kernel    
    
    Parameters
    ----------
    L: int
        Size of L×L spatial domain
    σ: positive number
        kernel radius exp(-x²/σ) 
        (standard deviation in x and y ×⎷2)
    normalize: boolean; default False
        whether to make kernel sum to 1
    '''
    k = blurkernel(L,σ)
    k = np.outer(k,k)
    if normalize: 
        k /= np.sum(k)
    return k

def conv(x,K):
    '''
    Compute circular 2D convolution using FFT
    Kernel K should already be fourier-transformed
    
    Parameters
    ----------
    x: 2D array
    K: Fourier-transformed convolution kernel
    '''
    x = np.array(x)
    return fft.ifft2(fft.fft2(x.reshape(K.shape))*K).real

def blur(x,σ,**kwargs):
    '''
    2D Gaussian blur via fft
    
    Parameters
    ----------
    x: 2D np.array
    σ: float
        kernel radius exp(-x²/σ) 
        (standard deviation in x and y ×⎷2)
    '''
    kern = fft.fft(blurkernel(x.shape[0],σ,**kwargs))
    return conv(x,np.outer(kern,kern))
    
def findpeaks(q,height_threshold=-np.inf,r=1):
    '''
    Find points higher than height_threshold, that are also 
    higher than all other points in a radius r circular
    neighborhood.
    
    Parameters
    ----------
    q: np.float32
        2D array of potential values
        
    Other Parameters
    ----------------
    height_threshold: float
        Peaks must be higher than this to cound.
    r: int
        Peaks must be larger than all other pixels in radius
        `r` to count.
        
    Returns
    -------
    np.bool
        2D boolean array of the same sape as q, indicating
        which pixels are local maxima within radius `r`.
    '''
    q  = np.array(q)
    L  = q.shape[0]
    D  = 2*r
    
    # Add padding
    Lpad = L+D;
    qpad = np.zeros((Lpad,Lpad)+q.shape[2:],dtype=q.dtype)
    qpad[r:-r,r:-r,...] = q[:,:,...]

    # Points to search
    Δ = range(-r,r+1)
    limit = r*r
    search = {(i,j) 
              for i in Δ 
              for j in Δ 
              if (i!=0 or j!=0) and (i*i+j*j)<=limit}
    
    # Only points above the threshold are candidate peaks
    p = q>height_threshold
    
    # Mask away points that have a taller neighbor
    for i,j in search:
        p &= q>qpad[i+r:L+i+r,j+r:L+j+r,...]
    
    return p

def dx_op(L):
    '''
    Parameters
    ----------
    L: int
        Size of L×L spatial grid
    '''
    # 2D difference operator in the 1st coordinate
    dx = np.zeros((L,L))
    dx[0, 1]=-.5
    dx[0,-1]= .5
    return dx

def hessian_2D(q):
    '''
    Get Hessian of discrete 2D function at all points
    
    Parameters
    ----------
    q: np.complex64
        List of peak locations encoded as x+iy complex
    '''
    q   = np.array(q)
    dx  = dx_op(q.shape[0])
    f1  = fft.fft2(dx)
    f2  = fft.fft2(dx.T)
    d11 = conv(q,f1*f1)
    d12 = conv(q,f2*f1)
    d22 = conv(q,f2*f2)
    return np.array([[d11,d12],[d12,d22]]).transpose(2,3,0,1)

def circle_mask(nr,nc):
    '''
    Zeros out corner frequencies
    
    Parameters
    ----------
    nr: int
        number of rows in mask
    nc: int
        number of columns in mask
    '''
    r = (np.arange(nr)-(nr-1)/2)/nr
    c = (np.arange(nc)-(nc-1)/2)/nc
    z = r[:,None]+c[None,:]*1j
    return abs(z)<.5

def fft_upsample_2D(x,factor=4):
    '''
    Upsample 2D array using the FFT
    
    Parameters
    ----------
    x: 2D np.float32
    '''
    if len(x.shape)==2:
        x = x.reshape((1,)+x.shape)
    nl,nr,nc = x.shape
    f = fft.fftshift(fft.fft2(x),axes=(-1,-2))
    f = f*circle_mask(nr,nc)
    nr2,nc2 = nr*factor,nc*factor
    f2 = np.complex128(np.zeros((nl,nr2,nc2)))
    r0 = (nr2+1)//2-(nr+0)//2
    c0 = (nc2+1)//2-(nc+0)//2
    f2[:,r0:r0+nr,c0:c0+nc] = f
    x2 = np.real(fft.ifft2(fft.fftshift(f2,axes=(-1,-2))))
    return np.squeeze(x2)*factor**2

def interpolate_peaks(
    z,
    r=1,
    return_coordinates='index',
    height_threshold=None):
    '''
    Obtain peak locations by quadratic interpolation.
    
    Presently, this function expects a square array. 
    
    Parameters
    ----------
    z: ndarray, L×L×NSAMPLES
        A 3D array of sampled 2D grid-fields,
        where the LAST axis is the sample numer. 
        
    Other Parameters
    ----------------
    r: positive int; default 1
        Radius over which point must be local maximum to 
        include. 
        Defaults to 1 (nearest neighbors)
    return_coordinates: str; deftault 'index'
        Can be 'index' or 'normalized'. 
        If 'index', points will be returns in units of 
        array index.
        If 'normalized', points will be returned in 
        units [0,1]².
    height_threshold: float
        Threshold (in height) for peak inclusion. 
        Defaults to the 25th percentil of z
        
    Returns
    -------
    ix: interpolated x location of peaks
    iy: interpolated y location of peaks
    rz: indecies of which sample each peak comes from
    '''
    z  = np.array(z)
    Lx = z.shape[0]
    Ly = z.shape[1]
    is3d = True
    if len(z.shape)==2:
        z = z.reshape(Lx,Ly,1)
        is3d = False
    if height_threshold is None:
        height_threshold=np.nanpercentile(z,25)
    
    # Peaks are defined as local maxima that are larger than all other
    # points within radius `r`, and also higher than the bottom 25% of
    # log-rate values. 
    peaks = findpeaks(z,height_threshold,r)
    # Local indecies of local maxima
    rx,ry,rz = np.where(peaks)
    # Use quadratic interpolation to localize peaks
    clipx = lambda i:np.clip(i,0,Lx-1)
    clipy = lambda i:np.clip(i,0,Ly-1)
    rx0 = clipx(rx-1)
    rx2 = clipx(rx+1)
    ry0 = clipy(ry-1)
    ry2 = clipy(ry+1)
    s00 = z[rx0,ry0,rz]
    s01 = z[rx0,ry ,rz]
    s02 = z[rx0,ry2,rz]
    s10 = z[rx ,ry0,rz]
    s11 = z[rx ,ry ,rz]
    s12 = z[rx ,ry2,rz]
    s20 = z[rx2,ry0,rz]
    s21 = z[rx2,ry ,rz]
    s22 = z[rx2,ry2,rz]
    dx  = (s21 - s01)/2
    dy  = (s12 - s10)/2
    dxx = s21+s01-2*s11
    dyy = s12+s10-2*s11
    dxy = (s22+s00-s20-s02)/4
    det = 1/(dxx*dyy-dxy*dxy)
    ix  = (rx-( dx*dyy-dy*dxy)*det + 0.5)
    iy  = (ry-(-dx*dxy+dy*dxx)*det + 0.5)
    # Rarely, ill-conditioning leads to inappropriate interpolation
    # We remove these cases. 
    bad = (ix<0) | (ix>Lx-1) | (iy<0) | (iy>Ly-1)
    ix  = ix[~bad]
    iy  = iy[~bad]
    if return_coordinates=='normalized':
        ix /= Lx
        iy /= Ly
    return np.float32((iy,ix,rz[~bad]) if is3d else (iy,ix))

def bin_spikes(px,py,s,L,w=None):
    '''
    Bin spikes, using linear interpolation to distribute 
    point mass to four nearest pixels, weighted by distance. 
    
    Parameters:
        px (np.flaot32): x location of points
        py (np.float32): y location of points
        s (np.array): spike count at each point
        L (int): number of spatial bins for the LxL grid
        w (np.float32): weights to apply to each point 
            (default is None for no weigting)
    '''
    # Bin spike counts simple version
    #N=histogram2d(y,x,(bins,bins),density=0,weights=w)[0]
    #ws=s if w is None else array(s)*array(w)
    #K=histogram2d(y,x,(bins,bins),density=0,weights=ws)[0]
    #return N,K
    
    if w is None:
        w = np.ones(len(px))
        
    assert np.max(px)<1 and np.max(py)<1 \
        and np.min(px)>=0 and np.min(py)>=0
    ix,fx = divmod(px*L,1)
    iy,fy = divmod(py*L,1)
    
    assert np.max(ix)<L-1 and np.max(iy<L-1)
    
    w11 = fx*fy*w
    w10 = fx*(1-fy)*w
    w01 = (1-fx)*fy*w
    w00 = (1-fx)*(1-fy)*w
    
    qx  = np.concatenate([ix,ix+1,ix,ix+1])
    qy  = np.concatenate([iy,iy,iy+1,iy+1])
    z   = np.concatenate([w00,w10,w01,w11])
    
    ibins = arange(L+1)
    N   = np.histogram2d(qy,qx,(ibins,ibins),
                      density=False,weights=z)[0]
    ws  = z*concatenate((s,)*4)
    K   = np.histogram2d(qy,qx,(ibins,ibins),
                      density=0,weights=ws)[0]
    
    return np.float32(N),np.float32(K)

def get_peak_density(z,resolution,r=1,height_threshold=None):
    '''
    Obtain peaks by quadratic interpolation, then bin
    the results to a spatial grid with linear interpolation.
    
    Parameters
    ----------
    z: ndarray, L×L×NSAMPLES
        A 3D array of 2D grid field samples,
        where the LAST axis is the sample numer. 
    resolution: int>1
        Upsampling factor for binned peal locations
        
    Other Parameters
    ----------------
    r: integer
        Radius over which point must be local maximum to 
        include. 
        Defaults to 1.
    height_threshold: float
        Threshold (in height) for peak inclusion. 
        Defaults to the 25th percentil of z
    '''
    L = z.shape[0]
    # Get list of peak locations
    iy,ix = interpolate_peaks(z,
        r=r,
        height_threshold=height_threshold)[:2]
    # Bin peaks on a (possibly finer) spatial grid with
    # linear interpolation. 
    return bin_spikes(iy,ix,0*iy,L*resolution)[0]

def brute_force_local_2d_maxima(x,R=5):
    '''
    Find points higher than all neighbors within 
    radius r in a 2D array. 
    
    Parameters
    ----------
    x: 2D np.array; 
    Signal in which to locate local maxima.
    R: int; 
        RxR region in which a peak must be a local maxima to 
        be included.
    
    Returns
    -------
    (x,y): tuple of np.int32 arrays with peak coordinates.
    '''
    R   = int(np.ceil(R))
    pad = 2*R
    x   = np.array(x)
    h,w = x.shape
    
    padded = np.zeros((h+pad,w+pad))
    padded[R:-R,R:-R] = x
    
    best = np.full(x.shape,-np.inf)
    RR = R*R
    for dr in np.arange(-R,R+1):
        for dc in np.arange(-R,R+1):
            if dr==dc==0: continue
            if dr*dr+dc*dc>RR: continue
            best = np.maximum(
                best, 
                padded[R+dr:R+dr+h,R+dc:R+dc+w])
    
    ispeak = x>=best
    return np.where(ispeak)[::-1]
