#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Matrix routines

This class collects routines that operate on lists of lists. Typically,
arguments are in the form of a row-major ordered matrix, as well as 
the number of rows and number of elements in each row. These algorithms
tend to parallelise over rows, but not within rows. Typical algorithms
will have complexity proportional to the complexity of the corresponsing
serial algorithm operating on a single row.

In interest of fixing convention, GPU matricies shall be accepted as a
tuple of (data,cols). The number of rows is inferred from the length
of the data. 

TODO : check correlation matrix funcions, something is off here
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

from neurotools.obsolete.gpu.cpu.util import *
from neurotools.obsolete.gpu.cl.function import *
    
try:
    from pytools import memoize
except:
    print('Please install the pytools module')
    print('Attempting fallback to neurotools')
    from neurotools.jobs.ndecorator import memoize

gputranspose = lambda rows,cols:gpumap('x[(i%'+str(rows)+')*'+str(cols)+'+(i/'+str(rows)+')]')
'''Prepares a map kernel that transposed a row-major packed float matrix/ Eg gputranspose(rows,cols)(data) will transpose data. Creates a new, memoized, kernel for each array dimension'''

transpose = lambda m:(lambda a,b:cut(cpu(gputranspose(b,a)(gpufloatmat(m))),b))(len(m[0]),len(m))
'''This is a list datatype wrapper to gputranspose. It accepts a matrix as a list of lists, and returns the same form'''

class GPUMatrix:
    '''This is a shallow wrapper of GPUArray. A GPUMatrix is simply a 
    GPUArray containing the matrix in row major order, as well as the
    dimensions of the matrix. GPUArray might even already have this 
    functionality'''
    def __init__(self,data,rows,cols):  
        self.data=data
        self.rows=rows
        self.cols=cols
        self.matrix=cut(data,cols)

@memoize
def matkern(source):
    '''This is a higher order function to simplify row-parallelized
    matrix kernel creation. We assume that we have a kernel that accepts
    data, cols. We create a function that accepts data,cols,
    as either two arguments or a single tuple. We execute the kernel, 
    assuming that the return data is placed in the argument array. We
    return a tuple of the now modified data and the row length'''
    source = 'float *in = &data[n*tid];'+source
    kern = lambda:kernel('float *data, int n',source)
    def call(data,cols=None):
        if cols==None: 
            data,cols = data             
        kern()(len(data)/cols)(data,cols)
        return (data,cols)
    return call
    
@memoize
def matscalar(source):
    '''
    For creation of matrix kernels that compute scalar results. 
    Accepts source. Returns a function from (data,cols)->(scalars).
    '''
    source = 'float *in = &data[n*tid];'+source
    kern = lambda:kernel('float *data, int n, float *out',source)
    def call(data,cols=None):
        if cols==None:
            data,cols = data
        t=len(data)/cols
        out = gpuarray.zeros((t),np.float32)        
        kern()(t)(data,cols,out)
        return out
    return call
    
@memoize
def matouter(source):
    '''
    '''
    kern = lambda:kernel('float *data, int cols, int rows, float *out',source)
    def call(data,cols=None):
        if cols==None:
            data,cols = data
        t=len(data)/cols
        out = gpuarray.zeros((t*t),np.float32)
        kern()(t)(data,cols,t,out)
        return out
    return call
    
convertToZScores = matkern('''
    float mean = 0.0f;
    for (int j=0; j<n; j++) mean+=in[j];
    mean/=n;
    float dev=0;
    for (int j=0; j<n; j++) {
        float d = in[j]-mean;
        dev+=d*d;
    }
    dev=1.0f/sqrt(dev/n);
    for (int j=0; j<n; j++) in[j]=(in[j]-mean)*dev;
    ''')
'''
Equivalent to mean centering then normalization. This function does
not return a value, but replaces the contents of the given data.
'''
    
meanCenter = matkern('''
    float mean = 0.0f;
    for (int j=0; j<n; j++) mean+=in[j];
    mean/=n;
    for (int j=0; j<n; j++) in[j]=(in[j]-mean);
    ''')
'''
This will subtract the mean from each row. This function modifies its
arguments, replacing them with return values
'''
    
normalize = matkern('''
    float mag = 0.0f;
    for (int j=0; j<n; j++) mag+=in[j]*in[j];
    mag=1.0f/sqrt(mag);
    for (int j=0; j<n; j++) in[j]=in[j]*mag;
    ''')
'''
This will normalize each row of a matrix on parallel on the GPU
'''
    
magnitudes = matscalar('''
    float mag = 0.0f;
    for (int j=0; j<n; j++) mag+=in[j]*in[j];
    out[tid]=sqrt(mag);
    ''')
'''
This will return the magnitude of each row
'''
    
sums = matscalar('''
    float sum = 0.0f;
    for (int j=0; j<n; j++) sum+=in[j];
    out[tid]=sum;
    ''')
'''
This will return the sum of each row
'''

means = matscalar('''
    float mag = 0.0f;
    for (int j=0; j<n; j++) mag+=in[j];
    out[tid]=mag/n;
    ''')
'''
This will return the population mean for each row
'''

variances = matscalar('''
    float mean = 0.0f;
    for (int j=0; j<n; j++) mean+=in[j];
    mean/=n;
    float dev=0;
    for (int j=0; j<n; j++) {
        float d = in[j]-mean;
        dev+=d*d;
    }
    out[tid]=dev/n
    ''')
'''
This will return the population variance for each row
'''

samplevariances = matscalar('''
    float mean = 0.0f;
    for (int j=0; j<n; j++) mean+=in[j];
    mean/=n;
    float dev=0;
    for (int j=0; j<n; j++) {
        float d = in[j]-mean;
        dev+=d*d;
    }
    out[tid]=dev/(n-1)
    ''')
'''
This will return the sample variance for each row
'''

stds = compose(gpumap("sqrt($)"))(variances)
'''This will return the population standard deviation for each row'''

sstds = compose(gpumap("sqrt($)"))(samplevariances)
'''This will return the sample standard deviation for each row'''

dotproducts = matouter('''
    const int I = tid/cols;
    const int J = tid%cols;
    if (I<cols && J<cols && I<=J) {
        float *vi = &data[I*rows];
        float *vj = &data[J*rows];
        float sum=0.0;
        for (int i=0;i<rows;i++) 
            sum+=vi[i]*vj[i];
        output[I*cols+J]=sum;
        output[J*cols+I]=sum;
    }''')
'''Also known as : a matrix times its transpose. Input data is not
altered'''

correlation = compose(dotproducts)(convertToZScores)
'''
Computes mean centered correlation matrix from a list of vectors
'''
    
correlation2 = compose(dotproducts)(normalize)
'''
Computes the uncentered correlation matrix from a list of vectors
'''


    
