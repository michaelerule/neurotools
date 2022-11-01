#!/usr/bin/python
# -*- coding: UTF-8 -*-
# BEGIN PYTHON 2/3 COMPATIBILITY BOILERPLATE
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

try:
    xrange
except:
    xrange = range

'''
Contains higher order functions to make creation of GPU functions more 
succinct and compact. Also contains generic routines for manipulating CUDA 
source objects.
'''
try:
    import pycuda.gpuarray as gpuarray
    import pycuda.cumath as cumath
    from pycuda.elementwise import ElementwiseKernel
    from pycuda.compiler import SourceModule
except:
    import sys
    def missing(*args,**kwargs):
        if 'sphinx' in sys.modules:
            print('Please locate and install the pycuda GPU library')
        else:
            raise ValueError('Please locate and install pycuda GPU library')
    # TODO: shadow missing function with the above, which raises an error?

try:
    from pytools import memoize
except:
    print('Please install the pytools module')
    print('Attempting fallback to neurotools')
    from neurotools.jobs.ndecorator import memoize

from math import log
import re
import numpy as np 

from neurotools.obsolete.gpu.cpu.util import *
from neurotools.obsolete.gpu.cu.device import *

##############################################################################
# Source Code Utility Functions
##############################################################################

def format(code):
    '''
    This is a kernel source auto-formatter. It mostly just does auto-indent
    '''
    code = re.compile(r'//').sub(r'@',code)
    code = re.compile(r'^([^@\n]*)@([\n]*)\n').sub(r'@\2\n\1\n',code)
    code = re.compile(r'@').sub(r'//',code)
    code = re.compile(r'//([^\n]*)\n').sub(r'/*\1*/\n',code)
    code = re.compile(r'[\n\t ]+').sub(' ',code)
    code = re.compile(r';[\n\t ]*').sub('; ',code)
    code = re.compile(r';+').sub(';',code)
    code = re.compile(r';').sub(';\n',code)
    code = re.compile(r'[ ]*else[ ]*\{[ ]*\}[ ]*').sub(' ',code)
    code = re.compile(r'\{').sub('\n {\n',code)
    code = re.compile(r'\}').sub('}\n',code)
    code = re.compile(r'for[ ]*\(([^;]*)\n*;\n*([^;]*)\n*;\n*').sub(r'for(\1;\2;',code)
    code = re.compile(r'\*/').sub('\n',code)
    code = re.compile(r'/\*').sub('//',code)
    code = re.compile(r'^[ \t]*').sub('',code)
    code = re.compile(r'//([^\n]*)\n').sub(r'',code)
    newcode = ''
    indents = 0
    for line in code.split('\n'):
        indents -= len(re.compile(r'\}').findall(line))
        for i in xrange(0,indents):
            newcode += '    '
        indents += len(re.compile(r'\{').findall(line))
        newcode += line+'\n'
    return newcode

def printKernel(code):
    '''
    This prints out a kernel source with line numbers
    '''
    code = format(code)
    code = code.split('\n')
    labeldigits = ceil(log(len(code))/log(10))
    formatstring = "%0"+str(labeldigits)+"d %s"
    for i,line in enumerate(code):
        print(formatstring%(i+2,line))

##############################################################################
# GPU function generting metafunctions
##############################################################################
        
@memoize
def gpubin(fun):
    '''This is a small wrapper to simplify calling binary r = a op b kernels. It automates creation of the result array'''
    def ll(a,b):
        r=gpuarray.empty_like(a)
        fun(a,b,r)
        return r
    return ll

gpuscalar=gpubin
    
@memoize
def gpumap(exp):
    '''
    This is a small wrapper to simplify creation of b[i] = f(a[i]) map 
    kernels. The map function is passed in as a string representing a CUDA 
    expression. The dollar sign $ should denote the argument variable. A 
    return array is automatically constructed. For example, `gpumap('$')` 
    creates a clone or idenitiy kernel, so `A=gpumap('$')(B)` will assign a 
    copy of B to A. As a nontrivial example, a nonlinear map might function 
    could be created as `gpumap('1/(1+exp(-$))')`
    '''
    exp = "z[i]="+expsub(exp)
    map_kern = lambda:ElementwiseKernel("float *x, float *z",exp,"map_kern")
    def f(v):
        r=gpuarray.empty_like(v)
        map_kern()(v,r)
        return r
    return f

@memoize
def gpuintmap(exp):
    '''This is the same thing as gpumap except for integer datatypes'''
    exp = "z[i]="+expsub(exp)
    map_kern = lambda:ElementwiseKernel("int *x, int *z",exp,"map_kern")
    def f(v):
        r=gpuarray.empty_like(v)
        map_kern()(v,r)
        return r
    return f
    
expsub = lambda exp:re.compile(r'\$').sub(r'x[i]',exp)  

@memoize
def gpumapeq(exp):
    '''This is a small wrapper to simplify creation of a[i] = f(a[i]) map 
    kernels. The map function is passed in as a string representing a CUDA 
    expression. The dollar sign $ should denote the argument variable. The
    result is assigned into the original array, so no new memory is 
    allocated. For example, gpumap('$') 
    creates a clone or idenitiy kernel, so A = gpumap('$')(B) will assign a 
    copy of B to A. As a nontrivial example, a nonlinear map might function 
    could be created as gpumap('1/(1+exp(-$))')'''
    exp = expsub("$="+exp)
    map_kern = lambda:ElementwiseKernel("float *x",exp,"map_kern")  
    def f(v):
        map_kern()(v)
        return v
    return f

"""
@memoize
def gpuparametermap(exp):
    '''Similar to gpumap, except that the resulting obect accepts an 
    additional parameter list. I had to do this because I found I was
    implementing maps with parameters, like log(x+c) with the parameter hard
    compiled in, which was somewhat inefficient. At this point you may be 
    wondering why I'm not just using ElementWiseKernel. Anyway, 
    gpuparametermap(expression) returns a curried function that first
    accepts a parameter list, then the data. The map expession should 
    indic'''
    exp = expsub("$="+exp)
    print(exp)
    map_kern = lambda:ElementwiseKernel("float *x",exp,"map_kern")  
    def f(v):
        map_kern()(v)
        return v
    return f
"""

@memoize  
def gpubinaryeq(exp):
    '''
    This wrapper simplified the creation of kernels executing operators
    like `{'+=','-=','*=','/='}`. That is, binary operators that assign the
    result to the left operator. This is to suppliment the functionality of
    PyCUDA GPUArrays, which support binary operations but always allocate a 
    new array to hold the result. This wrapper allows you to efficiently 
    execute binary operations that assign the result to one of the argument
    arrays. For example, implement the GPU equivalent of `+=` as 
    `gpubinaryeq('$x+$y')(x,y)`. The result will automatically be assigned to
    the first argument, x.
    '''
    exp = "$x="+exp
    exp = (lambda exp:re.compile(r'\$x').sub(r'x[i]',exp))(exp)
    exp = (lambda exp:re.compile(r'\$y').sub(r'y[i]',exp))(exp)
    map_kern = lambda:ElementwiseKernel("float *x, float *y",exp,"map_kern")  
    def f(v,w):
        map_kern()(v,w)
        return v
    return f
    
    
def guessGPUType(arg):
    '''At the moment, this returns numpy.float32 for Python floats and 
    numpy.int32 for python integers, and is otherwise undefined'''
    if arg.__class__==float:
        return np.float32
    elif arg.__class__==int:
        return np.int32
    else:
        return lambda x:x
    
    
toGPUType = lambda arg:guessGPUType(arg)(arg)
'''A little wrapper to auto-cast floats/ints to respective numpy datatypes
for use on the GPU. This functionality probably exists elsewhere'''
        
@memoize
def ezkern(header, code, other=None):
    '''
    This is my easy kernel wrapper. This function accepts a header ( the
    list of arguments ), a body ( the core of the loop ), and optionally
    a block of helper function code. The core loop should reference "tid" as
    the thread index variable. The distribution of threads on the GPU is 
    automatically managed.
    '''    
    source = """
    __global__ void fun(%(header)s, int ezkernelements, int ezkernstride) {
        const int istart = (blockIdx.x*blockDim.x+blockIdx.y)*blockDim.y+threadIdx.x;
        for (int tid=istart; tid<ezkernelements; tid+=ezkernstride) {
            %(code)s;
        }
    }"""%{'header':header, 'code':code}
    if other!=None:
        source = other+source
    printKernel(source)
    myModule = SourceModule(source)
    mykernel = myModule.get_function('fun')
    estimateThreadsPerBlock(myModule)
    @memoize
    def init(n_units):
        blocks      = estimateBlocks(myModule,n_units)
        myblock     = (myModule.threads_per_block,1,1)
        mygrid      = (myModule.blocks,1)
        otherargs   = [np.int32(n_units),np.int32(myModule.threads_per_block*myModule.blocks)]
        otherkwargs = {'block':myblock, 'grid':mygrid}
        def execkern(*args):
            a=cmap(toGPUType)(list(args))
            a.extend(otherargs)
            mykernel(*tuple(a),**otherkwargs)
            return
        return execkern
    return init
    
kernel     = ezkern


gpupointer = lambda gpuarr:gpuarr.gpudata
'''Returns the starting memory location of a GPUArray'''

cpu        = lambda v:v.get() 
'''Casts a gpu array to respective numpy array type'''


gpufloat   = lambda v:gpuarray.to_gpu((np.array(v)).astype(np.float32))
'''Casts a python list to a float array on the gpu'''

gpufloatmat= lambda M:gpu(flat(M))
'''Moves a python list of lists of floats to a GPU row major packed integer matric simply by flattening the python datastructure and copying'''

gpufloatred= lambda fun:lambda v:float((fun(v,np.float32)).get())
'''Wraps a GPUArray reduction function into a succint form operating on float arrays'''

gpuint     = lambda M:gpuarray.to_gpu(np.array(M).astype(np.int32))
'''Casts a python list to an integer array on the GPU'''
gpuintmat  = lambda M:gpuint(flat(M))

'''Moves a python list of lists of integers to a GPU row major packed integer matric simply by flattening the python datastructure and copying'''

gpuintred  = lambda fun:lambda v:float((fun(v,np.int32)).get())
'''Wraps a GPUArray reduction function into a succint form operating on int arrays'''




