'''
This is a collection of GPU and CPU code that I have written for data analysis within the PyCuda framework. I really shouldn't publish this because... well, its just ... really bad code for the most part. Not everything here is GPU based. Several Utilities are simply CPU based, or have not yet been ported to the GPU.

This is a collection of Python code written for data analysis using PyCuda. For the most part this project is a grab bag of random functions I have  written over the past few months. The code quality is not... great, but I hope to fill in the library over time.

released under GPL
written by Michael Rule
'''

import pycuda
import pycuda.driver as cuda
import pycuda.compiler
import pycuda.tools

cuda.init()

# We will use the current context if one is initialized, otherwise we create
# a new context.
#ctx = pycuda.tools.make_default_context()
dev = pycuda.tools.get_default_device()
ctx = dev.make_context()


