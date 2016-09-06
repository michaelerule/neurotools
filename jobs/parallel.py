#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
Parallel tools
==============

Notes on why "parmap" and related functions are awkward in Python:

Python multiprocessing will raise "cannot pickle"
errors if you try to pass general functions as arguments to parallel
map. This flaw arises because python multiprocessing is achieved by
forking subprocesses that contain a full copy of the interpreter state,
mapped in memory with copy-on-write. (incidentally this is also why
you don't want to load large datasets in worker processes, since each
process will load the dataset separately, consuming a large amount of
memory). Because functions defined after worker-pool initiation depend
on the interpreter state of the process in which they were defined, they
cannot be sent over to worker processes as we cannot guarantee in
general that the context they reference will exist in the worker process.
Consequentially, the only way to use a function with the "parmap"
function is to define it BEFORE the worker pool is initialized (or to
re-initialize the pool once it is defined). The function must be at
global scope.

Furthermore, the work-stealing pool model can only send back the return
values of function evaluations from the work queue -- and it does so in
no particular order. Therefore, if we want to know which job corresponds
to which return value, we must return identifying information. In this
case wer return the job number.

The reason we cannot make a generic function that masks this "return the
job ID" issue is that we cannot pass an arbitrary function over to
the worker pool processes due to the aforementioned interpreter context
issue.

A more laborous workaround, which we might consider later, would be to
re-implement the work-stealing queue so that job IDs are automatically
preserved and communicated through inter-process communication.

This will not solve the problem of needing to define all functions used
with "parmap" before the working pool is initailized and at global scope,
but it will save us from having to manually track and return the job
number, which will lead to more readable and more reusable code.
'''

from multiprocessing import Process, Pipe, cpu_count, Pool
import traceback, warnings
import sys
import signal
import threading
import functools

import inspect
import neurotools.jobs.decorator
import neurotools.jobs.closure

if sys.version_info<(3,0):
    from itertools import imap as map

__N_CPU__ = cpu_count()

reference_globals = globals()

def parmap(f,problems,leavefree=1,debug=False,verbose=False):
    global mypool
    problems = list(problems)
    njobs    = len(problems)

    if njobs==0:
        if verbose: print('NOTHING TO DO?')
        return []

    if not debug and (not 'mypool' in globals() or mypool is None):
        if verbose: print('NO POOL FOUND. RESTARTING.')
        mypool = Pool(cpu_count()-leavefree)

    enumerator = map(f,problems) if debug else mypool.imap(f,problems)
    results = {}
    sys.stdout.write('\n')
    for i,result in enumerator:
        sys.stdout.write('\rdone %0.1f%% '%((i+1)*100./njobs))
        sys.stdout.flush()
        if isinstance(result,tuple) and len(result)==1:
            result=result[0]
        results[i]=result
        if verbose and type(result) is RuntimeError:
            print('ERROR PROCESSING',problems[i])

    sys.stdout.write('\r            \r')
    return [results[i] if i in results else None \
        for i,k in enumerate(problems)]

def parmap_dict(f,problems,leavefree=1,debug=False,verbose=False):
    global mypool
    problems = list(problems)
    njobs    = len(problems)

    if njobs==0:
        if verbose: print('NOTHING TO DO?')
        return []

    if not debug and (not 'mypool' in globals() or mypool is None):
        if verbose: print('NO POOL FOUND. RESTARTING.')
        mypool = Pool(cpu_count()-leavefree)

    enumerator = map(f,problems) if debug else mypool.imap(f,problems)
    results = {}
    sys.stdout.write('\n')
    for key,result in enumerator:
        if isinstance(result,tuple) and len(result)==1:
            result=result[0]
        results[key]=result
        if verbose and type(result) is RuntimeError:
            print('ERROR PROCESSING',problems[i])

    sys.stdout.write('\r            \r')
    
    results = {key:results[key] for key in problems if key in results and not results[key] is None}
    return results

def parmap_indirect_helper(args):
    global reference_globals
    (lookup_mode,function_information,args) = args #py 2.5/3 compat.
    '''
    Multiprocessing doesn't work on functions that weren't accessible from
    the global namespace at the time that the multiprocessing pool was
    created.

    However, it is possible to send a subset of "safe" functions to
    worker processes. This can be done either by telling the worker
    where to find the function in system-wide libraries, or by sending
    over the function source code in the event that it does not depend on
    or close over mutable state.

    Multiprocessing can already map top level functions by name, but it only
    supports passing iterables to the map functions, which limit functions
    to one argument. This wrapper merely unpacks the argument list.
    '''
    try:
        if lookup_mode == 'module':
            # Locate function based on module name
            module,name = function_information
            return getattr(
                locals ().get(module) or
                globals().get(module) or
                __import__   (module),
                name)(*args)
        elif lookup_mode == 'source':
            # Regenerate function from source. RISKY
            source,name = function_information
            if not reference_globals is None:
                exec(source,reference_globals)
            else:
                exec(source)
            f = (locals ().get(name) or
                 globals().get(name))
            if f is None:
                if not reference_globals is None:
                    f = reference_globals[name]
            f.__source__ = source
            g = neurotools.jobs.decorator.unwrap(f)
            if not g is None:
                g.__source__ = source
            return f(*args)
        else:
            raise NotImplementedError(\
                "indirect call mode %s not implemented"%lookup_mode)
    except Exception as exc:
        # In event of an error, return error info
        return RuntimeError(traceback.format_exc(), exc)

def parmap_indirect(f,problems,leavefree=1,debug=False,verbose=False):
    '''
    Functions cannot be pickled. Multiprocessing routines must be defined
    in the global namespace before the pool is initialized, so that the
    child processes have access to the function definitions. However, under
    some restricted circumstances it may be possible to specify a function
    more dynamically, at runtime. This may be the case if:
        1   A function is part of a system-wide installed module
        2   A function can be "regenerated" from source code
    '''
    # don't allow closing over state that is likely to mutate
    # this isn't 100% safe, mutable globals remain an issue.
    neurotools.jobs.closure.verify_function_closure(f)

    name = f.func_name

    if f.__module__ != '__main__':
        # can say where to locate function.
        information = f.__module__,name
        lookup = 'module'
    else:
        # Function is not defined through a module. We might be able to
        # send the source code.
        print("Attempting to fall back on source code based solution")
        print("RISKY")
        source = inspect.getsource(neurotools.jobs.decorator.unwrap(f))
        information = source,name
        lookup = 'source'
    # parmap_indirect_helper(lookup_mode,function_information,args)
    results = parmap(\
        parmap_indirect_helper,
        [(lookup,information,args) for args in problems],
        leavefree=leavefree,
        debug=debug,
        verbose=verbose)
    # Detect failed jobs, assign them something nasty that's likely to
    # bet detected downstream.
    for i,(args,result) in enumerate(zip(args,results)):
        if type(result) is RuntimeError:
            print('Job %d %s failed'%(i,args))
            traceback, exception = result.args
            print(traceback)
        results[i] = NotImplemented
    return results

def reset_pool(leavefree=1,context=None):
    global mypool, reference_globals

    # try to see what the calling function sees
    if not context is None:
        reference_globals = context

    if not 'mypool' in globals() or mypool is None:
        print('NO POOL FOUND. STARTING')
        mypool = Pool(cpu_count()-leavefree)
    else:
        print('POOL FOUND. RESTARTING')
        print('Attempting to terminate pool, may become unresponsive')
        # http://stackoverflow.com/questions/16401031/python-multiprocessing-pool-terminate
        def close_pool():
            global mypool
            if not 'mypool' in globals() or mypool is None:
                return
            mypool.close()
            mypool.terminate()
            mypool.join()
        def term(*args,**kwargs):
            sys.stderr.write('\nStopping...')
            stoppool=threading.Thread(target=close_pool)
            stoppool.daemon=True
            stoppool.start()
        signal.signal(signal.SIGTERM, term)
        signal.signal(signal.SIGINT,  term)
        signal.signal(signal.SIGQUIT, term)
        del mypool
        mypool = Pool(cpu_count()-leavefree)

def parallel_error_handling(f):
    '''
    We can't really use exception handling in parallel calls.
    This is a wrapper to just catch errors to prevent them from
    propagating up.
    '''
    def parallel_helper(args):
        try:
            return f(*args)
        except Exception as exc:
            traceback.print_exc()
            info = traceback.format_exc()
            return RuntimeError(info, exc)
    return parallel_helper
