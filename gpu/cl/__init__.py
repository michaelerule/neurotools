"""
Here is the OpenCL implementation
"""

import pyopencl
import numpy

ctx   = pyopencl.Context()
queue = pyopencl.CommandQueue(ctx)
mf    = pyopencl.mem_flags


