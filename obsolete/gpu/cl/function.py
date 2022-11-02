#!/us#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Functions to make creation of GPU functions more 
succinct and compact. Also contains generic routines for manipulating Cl 
source objects.
'''
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

try:
    import pyopencl
except:
    import sys
    def missing(*args,**kwargs):
        if 'sphinx' in sys.modules:
            print('Please locate and install the pyOpenCL GPU library')
        else:
            raise ValueError('Please locate and install pyOpenCL GPU library')
    # TODO: shadow missing function with the above, which raises an error?
    pyopencl = None

from neurotools.obsolete.gpu.cl import *   

try:
    from pytools import memoize
except:
    print('Please install the pytools module')
    print('Attempting fallback to neurotools')
    from neurotools.tools import memoize

from math import log,ceil
import re
import numpy as np 

def format(code):
    '''
    This is a kernel source auto-formatter. 
    It mostly just does auto-indent
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

# Substitute __global before pointers
# pointers look like
# [something without commas, spaces, or *][space][star][something without commas, spaces, or *][optional whitespace][comma]
# [something without commas, spaces, or *][star][space][something without commas, spaces, or *][optional whitespace][comma]
insert_global = lambda s: re.compile(r'([^,* \t]+) \*([^,* \t]+)').sub(r'__global \1 *\2',s)


@memoize
def kernel(header, code, other=None):
    '''
    This is my easy kernel wrapper. This function accepts a header ( the
    list of arguments ), a body ( the core of the loop ), and optionally
    a block of helper function code. The core loop should reference "tid" as
    the thread index variable. The distribution of threads on the GPU is 
    automatically managed.
    '''
    source = """
    __kernel void fun(%(header)s) {
        const int gid = get_global_id(0);
        %(code)s;
    }"""%{'header':insert_global(header), 'code':code}
    if other!=None:
        source = other+source
    source=format(source)
    printKernel(source)
    source="""
    #pragma extension cl_khr_byte_addressable_store : enable
    #pragma extension cl_nv_compiler_options : enable
    #pragma extension cl_nv_device_attribute_query : enable
    #pragma extension cl_khr_global_int32_base_atomics : enable
    #pragma extension cl_khr_global_int32_extended_atomics : enable
    #pragma extension cl_khr_local_int32_base_atomics : enable
    #pragma extension cl_khr_local_int32_extended_atomics : enable
    """+source
    @memoize
    def mykernel(): return pyopencl.Program(ctx,source).build()
    @memoize
    def init(n_units):
        def execkern(*args): return mykernel().fun(queue, (n_units,), *args)
        return execkern
    return init

@memoize
def elemental(header, code):
    code=' '+code
    arrays = re.compile(r' \*([^,* \t]+)').findall(header)
    for var in arrays: 
        code = re.compile(r"([^a-zA-Z0-9_]+)%s([^a-zA-Z0-9_]+)"%var).sub(r"\1%s[gid]\2"%var,code)
    return kernel(header, code)
    #def execkern(*args):
    #    return kern(len(args[0]))(*args)
    #return execkern

def gpumap(source):
    print("gpumap under construction")

'''
def duckern(header,code):
    def get_type_string(a):
        try:
            return a._type_string
        except AttributeError:
            return None
    def duck(*args):
        types = map(args,get_type_string)
        print(types)
        
_assign_op = lambda t:lambda s:lambda a,b:elemental("%(T)sa,%(T)sb"%{T:t+" *"},"a%s=b;"%s)(len(a))(a,b)
_assign_op_float = _assign_op('float')
_assign_op_int = _assign_op('int')
_new_op = lambda t:lambda s:lambda a,b,c:elemental("%(T)sa,%(T)sb,%(T)sc"%{T:t+" *"},"c=a%sb;"%s)(len(a))(a,b,c)
_new_op_float = _v_op('float')
_new_op_int = _new_op('int')

sumeq =  _assign_op_float('+')  
difeq =  _assign_op_float('-')  
muleq =  _assign_op_float('*')  
diveq =  _assign_op_float('/')
  
sumeq =  _assign_op_float('+')  
difeq =  _assign_op_float('-')  
muleq =  _assign_op_float('*')  
diveq =  _assign_op_float('/')  
'''





