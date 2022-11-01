N = 50000

from orix.cl import *

a = numpy.random.rand(N).astype(numpy.int32)
a_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
dest_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
prg = pyopencl.Program(ctx, """
    #pragma extension cl_khr_byte_addressable_store : enable
    #pragma extension cl_nv_compiler_options : enable
    #pragma extension cl_nv_device_attribute_query : enable
    #pragma extension cl_khr_global_int32_base_atomics : enable
    #pragma extension cl_khr_global_int32_extended_atomics : enable
    #pragma extension cl_khr_local_int32_base_atomics : enable
    #pragma extension cl_khr_local_int32_extended_atomics : enable
    __kernel void foo(__global const int *a, __global int *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid];
      atom_inc(&c[gid]);
    }
    """).build()
prg.foo(queue, a.shape, a_buf, dest_buf)
inced = numpy.empty_like(a)
pyopencl.enqueue_read_buffer(queue, dest_buf, inced).wait()
if N != sum(inced-a):
    print ("simple kernel failed")
else:
    print("simple kernel test passed")

from orix.cl.function import *

a = np.random.rand(N).astype(np.int32)
a_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
dest_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
kernel("int *a, int *c","c[gid] = a[gid]; atom_inc(&c[gid]);")(len(a))(a_buf, dest_buf)
inced = np.empty_like(a)
pyopencl.enqueue_read_buffer(queue, dest_buf, inced).wait()
if N != sum(inced-a):
    print ("kernel wrapper failed")
else:
    print("easy kernel wrapper test passed")

a = np.random.rand(N).astype(np.int32)
a_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
dest_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
elemental("int *a, int *c","c = a; atom_inc(&c);")(len(a))(a_buf, dest_buf)
inced = np.empty_like(a)
pyopencl.enqueue_read_buffer(queue, dest_buf, inced).wait()
if N != sum(inced-a):
    print ("elementwise wrapper failed")
else:
    print("elementwise wrapper test passed")

from orix.cl.device import *

from orix.cl.matrix import *








