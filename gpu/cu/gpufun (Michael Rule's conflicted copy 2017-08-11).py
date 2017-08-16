'''
This module contains generically useful GPU clones of several simple functions. Most of these use the orix.function utility library to succinctly wrap GPU functions into python wrapped kernel calls. Note that elementwise c=a op b for op in {+,-,*,/,^} are supported as 
overloaded operators for gpuarrays and are not duplicated here. Much of this is somewhat useless wrapping of GPUArray and pycuda.cumath into other syntax without adding new functionality.
'''

import pycuda
import pycuda.gpuarray as gpuarray
from orix.cu.function import *
from orix.cpu.util import *
import numpy as np 
import pycuda.curandom

##############################################################################
# GPU functions
##############################################################################

'''silly little wrappers for things already in GPUArray or cumath'''
gpulcomb = lambda a,b,c,d:gpubin(lambda A,B,r:ElementwiseKernel(
    "float a, float *x, float b, float *y, float *z",
    "z[i] = a*x[i] + b*y[i]",
    "lin_comb")(c,A,d,B,r))(a,b)
'''Wraps a linear combination operator. 
gpulcomb(weight1,weight2,data1,data2) will return the elementwise linear 
combination weight1*data1[i]+weight2*data2[i]. Succesive calls do not 
cause recompiliation of the kernel'''
gpumean  = lambda v:gpusum(v)/float(len(v))
'''Average of GPU array'''

'''This module is a collection of GPU map kernels implementing common
functions not present in pycuda.cumath'''
gpunpdf  = lambda m,s:gpumap('%s*expf(%s*pow($-%s,2))'%(0.39894228/s,-0.5/(s*s),m))
'''Creates a normal distribution PDF elementwise evaluator. E.g. 
gpupdf(0,1) will create a zero-mean, unit standard deviation normal 
distribution. gpupdf(0,1)(data) will evaluate the PDF at all elements of 
data and return the results in a new array. New calls to gpupdf do cause 
compiliation of new kernel code, but kernels are memoized so a give 
(mean,standard_deviation) kernel will only be compiled once'''
gpulogpdf= lambda m,s:gpumap('%s+%s*pow($-%s,2)'%(log(0.39894228/s),-0.5/(s*s),m))
'''This creates an element-wise kernel evaluating the natural log of the 
PDF of a normal distribtion. E.g. gpulogpdf(0,1) creates an element-wise 
operator that evaluates the log of the probability for a zero-mean unit 
standard deviation normal distribution.'''
gpuhill  = lambda x:gpumap('$/($+%s)'%x)
'''Hill equation for noncooperative binding : f(x)=x/(x+c)''' 


'''This module contains functions for drawing random numbers from a
variety of distributions on the GPU'''
gpurandf   = lambda n:pycuda.curandom.rand(n)
'''Wrapper for pycuda.curandom.rand(n)'''
gpuuniform = lambda a,b:lambda n:gpurandf(n)*(b-a)+a
'''Curried GPU uniform random number generator. For example, 
gpuuniform(0,1) will create a function that returns uniform random 
numbers over [0,1). gpuuniform(0,1)(100) would create a GPU array of 100 
draws from a uniform [0,1) distribution'''
gpurandexp = lambda n:gpumap('log($)')(gpurandf(n))*(-1)
'''Generates exponentially distributed random numbers on the GPU'''

