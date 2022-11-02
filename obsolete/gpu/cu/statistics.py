#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Contains statistical routines. All routines assume float32 arrays as the
underlying datatype.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function


try:
    import pycuda.gpuarray
    from pycuda.elementwise import ElementwiseKernel
except:
    print('PyCuda is missing; please install it to use CUDA routines')
    pycuda = cuda = None
    def ElementwiseKernel(*args,**kwargs):
        print('PyCuda is missing; please install it to use CUDA routines')

from . import gpufun
from gpufun import *
from math import *
import numpy

'''This class collects routines that operate on a single float vector. 
These routines tend to use reductions in the computation and most have
complexity log(n) where n is the length of the vector'''


sdv_kern  = ElementwiseKernel(
    "float *x,float mean,float *z",
    "z[i]=pow(x[i]-mean,2)",
    "sdv_kern")


gpusdv   = gpuscalar(sdv_kern)
'''Computes elementwise squared deviation from a constant value. For
example, gpusdev(data,c) will return the squared distance of all elements
in data from c. This is a slightly better way of writing (data-c)**2 as
it avoids an intermediate array creation and copy'''


gpumean  = lambda v:gpusum(v)/float(len(v))
'''Computes the population mean of a float vector on the GPU'''


gpucenter= lambda v:gpushift(v,-gpumean(v))
'''Mean-centers a vector on the GPU'''


gpusqmag = lambda v:gpusum(v**2)
'''Computes the squared magnitude of a vector'''


gpumag   = compose(sqrt)(gpusqmag)
'''Computes the magnitude of a vector'''


gpusqdev = lambda v:gpusum(gpusdv(v,gpumean(v)))
'''Computes the sum of squared deviation from mean for a vector''' 


gpuvar   = lambda v:gpusqdev(v)/float(len(v))
'''Computes the population variance of a vector'''


gpusvar  = lambda v:gpusqdev(v)/(float(len(v))-1)
'''Computes the sample variance of a vector'''


gpustd   = compose(sqrt)(gpuvar)
'''Computes the population standard deviation of a vector'''


gpusstd  = compose(sqrt)(gpusvar)
'''Computes the sample standard deviation of a vector'''


gpucov   = lambda a,b:gpudot(gpucenter(a),gpucenter(b))/float(len(a))
'''Computes the covariance of two vectors.'''


gpucorr  = lambda a,b:gpucov(a,b)/(gpustd(a)*gpustd(b))
'''Computes the correlation coefficient between two vectors'''


gpuscov  = lambda a,b:gpudot(gpucenter(a),gpucenter(b))/float(len(a)-1)
'''Computes the sample covariance of two vectors'''


gpuscorr = lambda a,b:gpuscov(a,b)/(gpusstd(a)*gpusstd(b))
'''Computes the sample correlation of two vectors'''


gpusem   = lambda v:(gpusvar(v)/len(v))**0.5
'''Computes the standard error of mean for a vector'''


gpuzscore= lambda v:gpumul(1.0/gpustd(v))(gpucenter(v))
'''Computes the z-scores for a vector using sample statistics'''
   
   
##############################################################################
# A plotting helper
##############################################################################

gpubarlinekerna = ElementwiseKernel(
        "float *x, float low, float high, float *z",
        "z[i] = x[i]>=low&&x[i]<high?1.0:0.0",
        "gpubarlinekerna")
        
gpubarlinekernb = ElementwiseKernel(
        "float *p, float *x, float *z",
        "z[i]=p[i]>0?x[i]:0.0",
        "gpubarlinekernb")
        
gpubarlinekernc = ElementwiseKernel(
        "float *p, float *x, float mean, float *z",
        "z[i]=p[i]>0?pow(x[i]-mean,2):0.0",
        "gpubarlinekernc")
        
def gpubarlinedata(xdata,ydata,bins,minval=None,maxval=None):
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    if maxval==None: maxval=gpumax(xdata)
    if minval==None: minval=gpumin(xdata)
    binsize= (maxval-minval)/float(bins)
    inbin  = gpuarray.empty_like(xdata)
    select = gpuarray.empty_like(xdata)
    xmeans = []
    ymeans = []
    errors = []
    for i in xrange(bins):
        lo=minval+binsize*i;
        hi=minval+binsize*(i+1);
        gpubarlinekerna(xdata,lo,hi,inbin)
        N=gpusum(inbin)
        if N>1:
            gpubarlinekernb(inbin,ydata,select)
            my=gpusum(select)/float(N)
            gpubarlinekernb(inbin,xdata,select)
            mx=gpusum(select)/float(N)
            gpubarlinekernc(inbin,ydata,my,select)
            s=sqrt(gpusum(select)/(N*(N-1)))
            xmeans.append(mx)
            ymeans.append(my)
            errors.append(s)
    return (xmeans,ymeans,errors)    
    
def sebarline(datasets,bins,min=None,max=None,lx="",ly="",title=""):
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (x,y) in datasets:
        xm,ym,err=gpubarlinedata(x,y,bins,min,max)
        plt.errorbar(xm,ym,yerr=map(lambda x:2*x,err))
    ax.set_xlabel(textf(lx))
    ax.set_ylabel(textf(ly))
    ax.set_title(textf(title))
    fig.show()    
    
def sebarline2(datasets,lx="",ly="",title=""):
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (x,y) in datasets:
        ym=cmap(gpumean)(y)
        ys=cmap(gpusem)(y)*2
        plt.errorbar(x,ym,yerr=ys)
    ax.set_xlabel(textf(lx))
    ax.set_ylabel(textf(ly))
    ax.set_title(textf(title))
    fig.show()      
         
def gpuhistogram(xdata,ydata,bins,minval=None,maxval=None):
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    if maxval==None: maxval=gpumax(xdata)
    if minval==None: minval=gpumin(xdata)
    binsize= (maxval-minval)/float(bins)
    inbin  = gpuarray.empty_like(xdata)
    N = []
    for i in xrange(bins):
        gpubarlinekerna(xdata,minval+binsize*i,minval+binsize*(i+1),inbin)
        N.append(gpusum(inbin))
    return N


sdgpubarlinekerna = ElementwiseKernel(
        "float *x, float low, float high, float *z",
        "z[i] = x[i]>=low&&x[i]<high?1.0:0.0",
        "gpubarlinekerna")
sdgpubarlinekernb = ElementwiseKernel(
        "float *p, float *x, float *z",
        "z[i]=p[i]>0?x[i]:0.0",
        "gpubarlinekernb")
sdgpubarlinekernc = ElementwiseKernel(
        "float *p, float *x, float mean, float *z",
        "z[i]=p[i]>0?pow(x[i]-mean,2):0.0",
        "gpubarlinekernc")


def sdgpubarlinedata(xdata,ydata,bins,minval=None,maxval=None):
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    if maxval==None: maxval=gpumax(xdata)
    if minval==None: minval=gpumin(xdata)
    binsize= (maxval-minval)/float(bins)
    inbin  = gpuarray.empty_like(xdata)
    select = gpuarray.empty_like(xdata)
    xmeans = []
    ymeans = []
    errors = []
    for i in xrange(bins):
        lo=minval+binsize*i;
        hi=minval+binsize*(i+1);
        gpubarlinekerna(xdata,lo,hi,inbin)
        N=gpusum(inbin)
        if N>1:
            gpubarlinekernb(inbin,ydata,select)
            my=gpusum(select)/float(N)
            gpubarlinekernb(inbin,xdata,select)
            mx=gpusum(select)/float(N)
            gpubarlinekernc(inbin,ydata,my,select)
            s=sqrt(gpusum(select)/(N-1))
            xmeans.append(mx)
            ymeans.append(my)
            errors.append(s)
    return (xmeans,ymeans,errors)
        

def sdbarline(datasets,bins,min=None,max=None,lx="",ly="",title=""):
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (x,y) in datasets:
        xm,ym,err=sdgpubarlinedata(x,y,bins,min,max)
        plt.errorbar(xm,ym,yerr=map(lambda x:2*x,err))
    ax.set_xlabel(textf(lx))
    ax.set_ylabel(textf(ly))
    ax.set_title(textf(title))
    fig.show()    


def sdbarline2(datasets,lx="",ly="",title=""):
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (x,y) in datasets:
        ym=cmap(gpumean)(y)
        ys=cmap(gpusstd)(y)*2
        plt.errorbar(x,ym,yerr=ys)
    ax.set_xlabel(textf(lx))
    ax.set_ylabel(textf(ly))
    ax.set_title(textf(title))
    fig.show()

def gpubin_core(data,size):
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    cells      = int(len(data))
    timepoints = int(len(data[0]))
    data       = gpuint(flat(data))
    bins       = int(timepoints/size)
    n          = bins*cells
    newdata    = gpuarray.zeros(n,np.int32)
    kernel('int *destination, int *source, int size, int cells, int timepoints',\
    ''' const int newtimepoints = timepoints/size;
        const int cellID        = tid/newtimepoints;
        const int offset        = tid%newtimepoints;
        int *buffer             = &source[cellID*timepoints+offset*size];
        int sum = 0;
        for (int j=0; j<size; j++)
            sum += buffer[j];
        destination[tid]=sum;
    ''')(n)(newdata, data, np.int32(size), np.int32(cells), np.int32(timepoints))
    c=cpu(newdata)
    del newdata
    del data
    return cut(c,bins)
gpubin = lambda size:lambda data:flat([gpubin_core(data[i:i+MAXPROCESS],size) for i in xrange(0,len(data),MAXPROCESS)])


bin_code = """
__global__ void bin(float *source, int *dest, int N, int bins, float min, float max) {
    const int tid = __mul24(blockDim.x, blockIdx.x)+threadIdx.x;
    const int thN = __mul24(blockDim.x, gridDim.x);
    const float scale = 1.0f/(max-min)*bins;
    for (int idx = tid; idx < N; idx+=thN) {
        int bin = floor((source[idx]-min)*scale);
        if (bin<0) bin = 0;
        else if (bin>=bins) bin = bins-1;
        atomicAdd(&dest[bin],1);
    }
}
"""

try:
    gpu_bin = (pycuda.compiler.SourceModule(bin_code)).get_function("bin")
    gpu_bin.prepare([numpy.intp,numpy.intp,numpy.int32,numpy.int32,numpy.float32,numpy.float32],(256,1,1))
except Exception as exc:
    #import traceback
    #traceback.print_exc()
    print('PyCuda may not be installed, could not initialize')

def gpu_histogram(data,min,max,bins):
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    dest    = gpuarray.to_gpu(numpy.zeros((bins,),dtype=numpy.int32))
    source  = gpuarray.to_gpu(numpy.array(data,numpy.float32))
    sourcep = source.gpudata
    destp   = dest.gpudata
    N       = len(data)
    gpu_bin.prepared_call((ceil(N,256),1), 
        numpy.intp(sourcep), 
        numpy.intp(destp), 
        numpy.int32(N), 
        numpy.int32(bins), 
        numpy.float32(min), 
        numpy.float32(max))
    n = 1.0/N
    return [n*k for k in dest.get()]

def sprinkle(spikes,DT):
    '''
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    x,y = [],[]
    for t,spike in enumerate(spikes): 
        for i,s in enumerate(spike): 
            if s: 
                x.append(t*DT)
                y.append(i)
    return (x,y)

