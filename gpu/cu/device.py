'''Module orix.device contains functions that deal with things associated with
the physical graphics card device that I usually don't want to think about.'''


try:
    import pycuda.driver as cuda
except:
    import sys
    def missing(*args,**kwargs):
        if 'sphinx' in sys.modules:
            print('Please locate and install the pycuda GPU library')
        else:
            raise ValueError('Please locate and install pycuda GPU library')
    # TODO: shadow missing function with the above, which raises an error?
    
  
from neurotools.gpu.cpu.util import *

def estimateThreadsPerBlock(cudamodule):
    '''
    This function acceptas a cuda module. It will estimate the number of 
    threads from this module that can fit in one block in the current context.
    It will return the largest number of threads that do not exceed the
    amount of shared memory, registers, or the hard limit on threads per 
    block, rounded down to a multiple of the warp size.
    '''
    regs = cuda.Device.get_attribute(cuda.Context.get_device(),cuda.device_attribute.MAX_REGISTERS_PER_BLOCK)
    smem = cuda.Device.get_attribute(cuda.Context.get_device(),cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
    warp = cuda.Device.get_attribute(cuda.Context.get_device(),cuda.device_attribute.WARP_SIZE)
    maxt = cuda.Device.get_attribute(cuda.Context.get_device(),cuda.device_attribute.MAX_THREADS_PER_BLOCK)
    cudamodule.threads_per_block = int(min(float(regs)/cudamodule.smem,maxt,float(smem)/cudamodule.registers)/warp)*warp
    #cudamodule.threads_per_block = int(min(float(regs)/cudamodule.shared_size_bytes,maxt,float(smem)/cudamodule.num_regs)/warp)*warp
    return cudamodule.threads_per_block

def estimateBlocks(cudamodule,n_units):
    '''
    Called after estimateThreadsPerBlock. 
    This function will estimate the number of blocks needed to run n_units. 
    It will not return more blocks than there are multiprocessors.
    
    If there are more blocks than multiprocessors, my convention is to loop
    within the kernel. It is unclear to me weather running more blocks than
    there are processors is more or less efficient than looping within blocks.
    '''
    mcount = cuda.Device.get_attribute(cuda.Context.get_device(),cuda.device_attribute.MULTIPROCESSOR_COUNT)
    cudamodule.blocks = min(mcount,ceil(float(n_units)/cudamodule.threads_per_block))
    return cudamodule.blocks

def estimateLoop(cudamodule,n_units):
    '''
    Called after estimateBlocks
    If there are not enough multiprocessors to handle n_units, this will
    return the number of loops within each kernel needed to process all data.
    '''
    cudamodule.loop = ceil(float(n_units)/(cudamodule.blocks*cudamodule.threads_per_block))
    return cudamodule.loop 
    
def card_info():
    '''
    returns information on the current GPU device as known to pycuda as a
    string
    '''
    result = ""
    properties = ["MAX_THREADS_PER_BLOCK",
    "MAX_BLOCK_DIM_X",
    "MAX_BLOCK_DIM_Y",
    "MAX_BLOCK_DIM_Z",
    "MAX_GRID_DIM_X",
    "MAX_GRID_DIM_Y",
    "MAX_GRID_DIM_Z",
    "TOTAL_CONSTANT_MEMORY",
    "WARP_SIZE",
    "MAX_PITCH",
    "CLOCK_RATE",
    "TEXTURE_ALIGNMENT",
    "GPU_OVERLAP",
    "MULTIPROCESSOR_COUNT",
    "MAX_SHARED_MEMORY_PER_BLOCK",
    "MAX_REGISTERS_PER_BLOCK"]
    for var in properties:
        result+="%s = "%var+"%d \n"%(cuda.Device.get_attribute(
            cuda.Context.get_device(),\
            cuda.device_attribute.__dict__[var]))
    return result





