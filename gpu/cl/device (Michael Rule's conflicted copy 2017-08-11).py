
from orix.cl import *
from orix.cl.function import *
import numpy as np
import pyopencl as cl

zero_device = elemental("int *d","d=0")

def zeros_float(N):
    arr = np.zeros(N);
    buf = cl.Buffer(ctx, mf.READ_WRITE|mf.USE_HOST_PTR, hostbuf=arr)
    return buf
