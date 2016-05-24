'''
This is a hobby project that probably should not be seen through to 
completion.


Note: nice features to add to the disk caching framework:
    -- a "touched" piece of information stating how recently a cache was
       used
    -- stats on how expensive a chunk is to compute
    
Note: cool things to create as decorators
    -- Decorator to detect whether function closes over mutable scope
       ( this includes mutable defaults )
    -- Creates EARLY BINDING behavior
    -- Objective: CREATE PICKLEABLE FUNCTIONS FOR DISTRIBUTED COMPUTING
       ( worse-case scenario the source code can be pickled and sent!)

I'm having some trouble with using disk-caching / disk-memoizing in python.
Namely, sometimes due to contention for disk IO, caching is slower than
recomputing. Since the level of contention varies at runtime, its hard
to predict in advance whether retrieving a cached copy is better / faster
than recomputing something. Furthermore, because CPU contention can 
also affect runtime, it's very hard to predict wether grabbing a disk
cache is faster or slower than recomputing on the CPU. 

Clearly, the answer here is to run BOTH approaches in parallel and see
which one returns faster, killing the slower once. This is perilous and 
must be done with extreme caution. 

No code yet, let's just sketch out the requirements: 

Functions involved should not close over mutable scope or have mutable
default arguments. 

Open design question: If fork() is used to implement this, then functions
need only exist in the current namespace at the time of invokation. If a
worker-pool is used to implement this, then the functions must exist
before said worker-pool is created, or somehow be serializable to send to
the worker processes.

Functions MAY perform disk IO. Primarily, this is to support calling of
routines to retrieve disk caches. It is OK to kill disk reads, it is NOT OK
to kill disk writes, as this will lead to needless proliferation of corrupt
caches. In a worst case scenario, a corrupted cache file may not even be
detectable! 

SO we need a SAFE way of killing processes, that can ensure that either
    [1] disk writing COMPLETES
or  [2] disk writing is ABORTED with NO FILE generated

[2] may be NECESSARY if there is high disk IO contention and we CANNOT wait
for cache writing of a partial computation to complete.

Getting all this right is HARD. Do it very slowly, and do thorough research
before you proceed.


'''
