#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

'''
This library contains a collection of kernels that I have found useful when
looking for repeating sequences in data
'''

def GPUSequenceAutoDetect(distances,t,thresh):
    '''
    '''
    out = gpuint(np.zeros(t*t))
    kernel('float *distances, int t, float thresh, int *out',
        '''
        int   ia    = tid%t;
        int   ib    = tid/t;
        int   l     = 0;
        int   stop  = min(t-ia,t-ib);
        int   index = tid;
        int   di    = 1+t;
        float s     = 0.0f;
        if (ia==ib) {
            out[tid]=0;
            return;
        }
        while (l<stop) {   
            float x=distances[index];
            s+=x;
            index+=di;
            l++;
            
            if (x>thresh) break;            
        }
        out[tid]=l;
        ''')(t*t)(distances,t,thresh,out)
    return out


def GPUPointAutoDistance(t,k,n,data):
    '''
    limit to 5-13k timestep for 256x256 sim
    
    Args:
        t (int):length of data in time
        k (int):number of time bins to use
        n (int):size of vector datapoints
        data (int):t*n data matrix, n is inner dimension
    '''
    w=t-k+1
    kern1=kernel('float *data, int n, int t, float *out',
        '''
        int outx = tid%t;
        int outy = tid/t;
        float *inx = &data[n*outx];
        float *iny = &data[n*outy];
        float sumsq = 0.0f;
        for (int j=0; j<n; j++) {    
            float a=inx[j]-iny[j];
            sumsq+=a*a;
        }
        out[tid]=sumsq/n;
        ''')(t*t)
    kern2=kernel('float *in, int k, int w, int t, float *out',
        '''
        int index = (tid%w)+(tid/w)*t;
        float sumsq = 0.0f;
        for (int i=0; i<k; i++) {   
            sumsq+=in[index];
            index+=1+t;
        }
        out[tid]=sqrt(sumsq/k);
        ''')(w*w)
    out  = gpu(np.zeros(t*t))
    out2 = gpu(np.zeros(w*w))
    kern1(data,n,t,out)
    kern2(out,k,w,t,out2)
    return (w,out2)

def GPUAutometric(t,n,data):
    '''
    limit to 5-13k timestep for 256x256 sim
    
    Args:
        t (int):length of data in time
        k (int):number of time bins to use
        n (int):size of vector datapoints
        data (int):t*n data matrix, n is inner dimension
    '''
    kern=kernel('float *data, int n, int t, float *out',
        '''
        int outx = tid%t;
        int outy = tid/t;
        float *inx = &data[n*outx];
        float *iny = &data[n*outy];
        float sumsq = 0.0f;
        for (int j=0; j<n; j++) {    
            float a=inx[j]-iny[j];
            sumsq+=a*a;
        }
        out[tid]=sqrt(sumsq/(n));
        ''')(t*t)
    out  = gpu(np.zeros(t*t))
    kern(data,n,t,out)
    return cpu(out)

def GPUMagmetric(t,n,data):
    '''
    limit to 5-13k timestep for 256x256 sim
    
    Args:
        t (int):length of data in time
        k (int):number of time bins to use
        n (int):size of vector datapoints
        data (int):t*n data matrix, n is inner dimension
    '''
    kern=kernel('float *data, int n, int t, float *out',
        '''
        float *inx = &data[n*(tid%t)];
        float *iny = &data[n*(tid/t)];
        float sum = 0.0f;
        for (int j=0; j<n; j++) sum+=inx[j]*iny[j];
        out[tid]=sum;
        ''')(t*t)
    out = gpu(np.zeros(t*t))
    kern(data,n,t,out)
    return cpu(out)

def GPUDotmetric(t,n,data):
    '''
    limit to 5-13k timestep for 256x256 sim
    
    Args:
        t (int):length of data in time
        k (int):number of time bins to use
        n (int):size of vector datapoints
        data (int):t*n data matrix, n is inner dimension
    '''
    kern=kernel('float *data, int n, int t, float *out',
        '''
        float *inx = &data[n*(tid%t)];
        float *iny = &data[n*(tid/t)];
        float sum = 0.0f;
        for (int j=0; j<n; j++) sum+=inx[j]*iny[j];
        out[tid]=sum;
        ''')(t*t)
    out = gpu(np.zeros(t*t))
    kern(data,n,t,out)
    return cpu(out)

def deltamag(t,data):
    kern=kernel('float *data, int t, float *out',
        '''
        out[tid]=data[tid%t]-data[tid/t];
        ''')(t*t)
    out = gpu(np.zeros(t*t))
    kern(data,t,out)
    return cpu(out)
    
def summag(t,data):
    kern=kernel('float *data, int t, float *out',
        '''
        out[tid]=data[tid%t]+data[tid/t];
        ''')(t*t)
    out = gpu(np.zeros(t*t))
    kern(data,t,out)
    return cpu(out)

def gpuderivative(i):
    t=len(i)-1
    o=gpu(np.zeros(t))
    kern=kernel('float *i, float *o','o[tid]=i[tid+1]-i[tid]')(t)(i,o)
    return o

def gpusmooth(rad,data):
    t=len(data)
    s=t-rad
    out=gpu(np.zeros(s))
    kernel('float *data,int r,float *out',
        '''
        float sum=0;
        for (int i=0; i<r; i++) sum+=data[tid+i];
        out[tid]=sum/r;
        ''')(s)(data,rad,out)
    return out

def gputhing(rad,data):
    t=len(data)
    s=t-rad
    out=gpu(np.zeros(s))
    norm = 1.0/sum([(i+.5)*(rad-(i+.5)) for i in xrange(rad)])
    kernel('float *data,int r,float norm,float *out',
        '''
        float sum=0;
        for (int i=0; i<r; i++){
            float w=i+0.5;
            w=i*(r-i);
            sum+=data[tid+i]*w;}
        out[tid]=sum*norm;
        ''')(s)(data,rad,norm,out)
    return out

def mulmag(t,data):
    kern=kernel('float *data, int t, float *out',
        '''
        out[tid]=data[tid%t]*data[tid/t];
        ''')(t*t)
    out = gpu(np.zeros(t*t))
    kern(data,t,out)
    return cpu(out)
    
def FrameEater(filename,maxframes=None):
    '''
    TODO: document
    
    file format : list of float images
    width height frames \n
    first frame as tab delimited floats on one line \n
    next frame \n
    and so on \n
    end file
    '''
    f = open(filename,'r')
    readhead = False
    frames = 0
    width  = None
    height = None
    data   = []
    castfloat = cmap(float)
    for line in f.xreadlines(): 
        if not readhead:
            header = line.split()
            width  = int(header[0])
            height = int(header[1])
            readhead = True
        else:
            frames=frames+1
            dataline=castfloat(line.split())
            data.extend(dataline)   
            print('read frame %s'%frames)
        if maxframes==None or frames>=maxframes:
            break 
    f.close()
    return (((width,height),frames),data)
    
def gpusubsetmean(p,d):
    n=gpusum(p)
    s=gpusum(p*d)
    return s/n

def gpusubsetgfit(p,d):
    n=gpusum(p)
    s=gpusum(p*d)
    mean = s/n
    d=gpusum(p*(d-mean)**2)
    sdv = sqrt(d/n)
    return mean,sdv,n
    
def gpunpdf(m,s,d):
    o=gpu(np.zeros(len(d)))
    kern=kernel('float *d, float *o, float c1, float c2, float c3',
    'o[tid]=c1*expf(c2*pow(d[tid]-c3,2))')(len(d))(d,o,0.39894228/s,-0.5/(s*s),m)
    return o
    
def gpulognpdf(m,s,d):
    o=gpu(np.zeros(len(d)))
    kern=kernel('float *d, float *o, float c1, float c2, float c3',
    'o[tid]=c1+c2*pow(d[tid]-c3,2)')(len(d))(d,o,log(0.39894228/s),-0.5/(s*s),m)
    return o

def fitgaussbimodal(gpupoints,steps):
    n=gpupoints
    mean=gpumean(n)
    ma,sa,ka=gpusubsetgfit(gpult(mean)(n),n)
    mb,sb,kb=gpusubsetgfit(gpugteq(mean)(n),n)
    isina=None
    isinb=None
    oldll=None
    ll=None
    for i in xrange(steps):
        pa=gpulognpdf(ma,sa,n)+log(ka)
        pb=gpulognpdf(mb,sb,n)+log(kb)
        agtb=(pa-pb)
        isina=gpugt(0)(agtb)
        isinb=gpunot(isina)
        ll=gpusum(isina*pa)+gpusum(isinb*pb)
        ma,sa,ka=gpusubsetgfit(isina,n)
        mb,sb,kb=gpusubsetgfit(isinb,n)
        if (oldll!=None):
            if ll<=oldll:
                print('bailed on iteration %s'%i)
                return ((ma,sa,ka,isina),(mb,sb,kb,isinb))  
        oldll=ll
    return ((ma,sa,ka,isina),(mb,sb,kb,isinb))
    
def fitgausstrimodal(gpupoints,steps):
    n=gpupoints
    mean=gpumean(n)
    dmin=gpumin(n)
    dmax=gpumax(n)
    c1=dmin+(dmax-dmin)/3
    c2=dmin+2*(dmax-dmin)/3
    ina=gpult(c1)(n)
    inb=gpurange(c1,c2)(n)
    inc=gpugteq(c2)(n)
    ma,sa,ka=gpusubsetgfit(ina,n)
    mb,sb,kb=gpusubsetgfit(inb,n)
    mc,sc,kc=gpusubsetgfit(inc,n)
    for i in xrange(steps):
        pa=gpunpdf(ma,sa,n)*ka
        pb=gpunpdf(mb,sb,n)*kb
        pc=gpunpdf(mc,sc,n)*kc
        agtb=pa-pb
        cgtb=pc-pb
        isina=gpugt(0)(agtb)
        isinc=gpugt(0)(cgtb)
        isinb=gpunot(gpunor(isina,isinc))
        ma,sa,ka=gpusubsetgfit(isina,n)
        mb,sb,kb=gpusubsetgfit(isinb,n)
        mc,sc,kc=gpusubsetgfit(isinc,n)
    return ((ma,sa,ka),(mb,sb,kb),(mc,sc,kc))


        
