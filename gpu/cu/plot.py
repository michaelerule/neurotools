'''Very rarely, I will want to accelerate a computation for plotting'''

try:
    from pycuda.elementwise import ElementwiseKernel
except:
    import sys
    def missing(*args,**kwargs):
        if 'sphinx' in sys.modules:
                print('Please locate and install PyCuda')
        else:
            raise ValueError('Please locate and install PyCuda')
    ElementwiseKernel = missing

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
    if maxval==None: maxval=gpumax(xdata)
    if minval==None: minval=gpumin(xdata)
    binsize= (maxval-minval)/float(bins)
    inbin  = gpuarray.empty_like(xdata)
    N = []
    for i in xrange(bins):
        gpubarlinekerna(xdata,minval+binsize*i,minval+binsize*(i+1),inbin)
        N.append(gpusum(inbin))
    return N
    
    
