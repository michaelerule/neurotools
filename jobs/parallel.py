#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
Parallel tools
==============
'''

from multiprocessing import Process, Pipe, cpu_count

import traceback
try:
    from multiprocessing import Pool
except ImportError as ie:
    print('Problem importing multiprocessing.Pool?')
    traceback.print_exc()

import traceback, warnings
import sys
import signal
import threading
import functools
import inspect
import neurotools.jobs.ndecorator
import neurotools.jobs.closure
from neurotools.tools import asiterable
import numpy as np

if sys.version_info<(3,0):
    from itertools import imap as map

__N_CPU__ = cpu_count()

reference_globals = globals()

def parmap(f,problems,leavefree=1,debug=False,verbose=False,show_progress=True):
    '''
    Parallel implmenetation of map using multiprocessing

    Parameters
    ----------
    f : function to apply, takes one argument and returns a tuple
        (i,result) where i is the index into the problems
    problems : list of arguments to f to evaulate in parallel
    leavefree : number of cores to leave open
    debug : if True, will run on a single core so exceptions get 
            handeled correctly
    verbose : set to True to print out detailed logging information
    
    Returns
    -------
    list of results
    '''
    global mypool

    problems = list(problems)
    njobs    = len(problems)
    if njobs==0: return []

    if not debug:
        try:
            mypool
        except NameError:
            if verbose: 
                print('No worker pool found, restarting.')
            mypool = Pool(cpu_count()-leavefree)

    enumerator = map(f,problems) if debug else mypool.imap(f,problems)
    results = {}
    lastprogress = 0.0
    thisprogress = 0.0
    try:
        for i,result in enumerator:
            if show_progress:
                thisprogress = ((i+1)*100./njobs)
                if (thisprogress - lastprogress)>0.5:
                    k = int(thisprogress//2)
                    sys.stdout.write('\r['+('#'*k)+(' '*(50-k))+']%5.1f%% '%thisprogress)
                    sys.stdout.flush()
                    lastprogress = thisprogress
            # if it is a one element tuple, unpack it automatically
            if isinstance(result,tuple) and len(result)==1:
                result=result[0]
            results[i]=result
            if verbose and type(result) is RuntimeError:
                print('Error processing',problems[i])
        if show_progress:
            sys.stdout.write('\r['+('#'*50)+']%5.1f%% \n\r'%100)
            sys.stdout.flush()
    except:
        if show_progress:
            k = int(thisprogress//2)
            sys.stdout.write('\r['+('#'*k)+('~'*(50-k))+'](fail)\n\r')
            sys.stdout.flush()
        raise
    return [results[i] if i in results else None \
        for i,k in enumerate(problems)]

def pararraymap(function,problems,debug=False):
    '''
    Parmap wrapper for common use-case with Numpy
    '''
    return np.array(parmap(function,enumerate(problems),debug=debug))

def parmap_dict(f,problems,leavefree=1,debug=False,verbose=False):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
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

"""
def parmap_indirect_helper(args):
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
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    global reference_globals
    (lookup_mode,function_information,args) = args #py 2.5/3 compat.
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
            g = neurotools.jobs.ndecorator.unwrap(f)
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
    
        1.   A function is part of a system-wide installed module
        2.   A function can be "regenerated" from source code
        
    Parameters
    ----------
    
    Returns
    -------
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
        source = inspect.getsource(neurotools.jobs.ndecorator.unwrap(f))
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
"""

"""
General plan for this: 

Handle these three casese 
1 : indirect parmap just works (somehow)
2 : try handling it by code
"""

# Each worker thread will end up maintaining its own copy of this
__indirect_eval_fallback_cache = {}

# Get name, source and hash of source
def function_fingerprint(f,verbose=False):
    nme = f.__code__.co_name
    src = inspect.getsource(f)
    nrg = len(inspect.getfullargspec(f).args)
    key = neurotools.jobs.cache.base64hash(src)
    if verbose:
        print('Name is:',nme)
        print('Hash is:',key[:30]+'...')
        print('Source is:')
        print('| '+src.replace('\n','\n| '))
    return nme,src,key,nrg
  
def eval_from_cached(fingerprint,args,verbose=False):
    '''
    Attempts to recompile a function from its source code, and store the
    compiled result in the global dictionary `__indirect_eval_fallback_cache`.
    It then attempts to call the function on the provided arguments.

    This is one alternative way to pass newly-created functions to the worker
    pool in parallel processing, if the usual built-in routines fail. 

    This can fail if the function closes over additional variables that were
    not present at the time the worker-pool was initialized. 

    It can yield undefined results if the function uses mutable variables in
    the global scope.

    This cannot rebuild lambda expressions from source, but can use them if 
    they are stored in the global __indirect_eval_fallback_cache ahead of time.
    '''
    global __indirect_eval_fallback_cache
    nme,src,key,nrg = fingerprint
    if not nrg==len(args):
        raise ValueError('Function %s expects %d args, but given %d'\
                         %(nme,nrd,len(args)))
    if key in __indirect_eval_fallback_cache:
        if verbose: print('Retrieved %s'%nme)
    else:
        if verbose: print('Rebuliding %s'%nme)
        if nme=='<lambda>':
            raise ValueError('Recompiling λ expressions unsupported')
        f_globals = dict(globals())
        if nme in f_globals: del f_globals[nme]
        f_locals = {}
        exec(src,f_globals,f_locals)
        __indirect_eval_fallback_cache[key] = f_locals[nme],src
    f = __indirect_eval_fallback_cache[key][0]
    return f(*args)

def parallel_indirect_wrapper(p):
    i,(f,args) = p
    return i,f(*args)

def parallel_cached_wrapper(p):
    i,(f,args) = p
    return i,eval_from_cached(f,args)

def __parimap_builtin(f,jobs,debug=False,verbose=False,show_progress=True):
    '''
    Attempt to use multiprocessing parallel map 'as directed'
    '''
    indirect_jobs = [(f,args) for args in jobs]
    parallel_indirect_wrapper((0,indirect_jobs[0]))
    return parmap(parallel_indirect_wrapper,
                  enumerate(indirect_jobs),
                            debug=debug,
                            verbose=verbose,
                            show_progress=show_progress)

from pickle import PicklingError

def parimap(f,jobs,
    debug=False,
    force_cached=False,
    force_fallback=False,
    allow_fallback=True,
    verbose=False,
    show_progress=True):
    '''
    Parallel map routine. 

    In some cases and on some systems, user-defind functions don't work with
    the multiprocessing library. 

    This happens when the system is unable to "pickle" functions which were
    defined after the worker pool was initiatlized.

    There are two workarounds for this: 

    (1) You can attempt to send the function source-code to the workers, and
        rebuild from source within the workers. However, this is risky for two
        reasons. (a) If your funcion closes over globals which were not defined
        at the time the worker-pool was launched, these globals will be missing
        and prevent re-compilation of the function. (b) Any mutable variables
        that the function closes over might have a different state in the
        worker threads (as I understand it). 

    (2) You can also ensure that there is a pointer to the defined funcion in
        the global workspace. Here, we store function pointers in the dictionary
        '__indirect_eval_fallback_cache'. Then, one can re-launch the worker
        pool, and each worker will gain access to the compiled function via the
        inhereted global scope (as I understand it). 

    If normal parallel map fails, this routine will first try (1), and then (2).
    '''
    global __indirect_eval_fallback_cache
    try:
        __indirect_eval_fallback_cache
    except NameError:
        __indirect_eval_fallback_cache = {}
    ############################################################################
    # check that we can get function argument signature
    if not hasattr(f, '__call__'):
        raise ValueError('1st argument to parimap must be callable.')
    try:
        nrg = len(inspect.getfullargspec(f).args)
        if verbose: print('Function takes %d arguments.'%nrg)
    except TypeError as te:
        if len(te.args)>0 and te.args[0]=='unsupported callable':
            raise ValueError('Could not identify # of function arguments')
    if nrg<1:
        raise ValueError('Functions with no arguments are not supported.')
    ############################################################################
    # Check that job list is well-formatted.
    jobs = asiterable(jobs)
    if jobs is None:
        raise ValueError('2nd argument to parimap must be iterable.')
    if len(jobs)==0: 
        raise ValueError('Empty job list passed to parimap.')
    ijobs      = [asiterable(j)  for j  in jobs ]
    iterstatus = [ij is not None for ij in ijobs]
    if nrg==1:
        if not all(iterstatus) or any([len(j)!=1 for j in ijobs]):
            jobs = [(j,) for j in jobs]
    elif not all(iterstatus):
        raise ValueError('All jobs must be an iterable with %d items'%nrg)
    elif not all([len(j)==nrg for j in ijobs]):
        raise ValueError('All jobs must specify %d arguments'%(nrg))
    ############################################################################
    # Try using built-in parallel map
    if not (force_cached or force_fallback):
        try:
            if verbose: print('Trying built-in parallel map.')
            result = __parimap_builtin(f,jobs,
                            debug=debug,
                            verbose=verbose,
                            show_progress=show_progress)
            if verbose: print('Built-in parallel map worked.')
            return result
        except (SystemExit,KeyboardInterrupt): raise
        except PicklingError:
            if verbose:
                print('Function could not be pickled.')
        except:
            if verbose: 
                print('Built-in parmap failed.')
                traceback.print_exc()
    ############################################################################
    # Send function to workers as source code
    fingerprint     = function_fingerprint(f)
    nme,src,key,nrg = fingerprint
    indirect_jobs   = [(fingerprint,args) for args in jobs]

    if nme=='<lambda>' and not key in __indirect_eval_fallback_cache:
        if verbose: print('Passing λ via source-code is not supported.')
    elif not force_fallback:
        try:
            eval_from_cached(fingerprint,jobs[0],verbose)
            parallel_cached_wrapper((0,indirect_jobs[0]))
            result = parmap(parallel_cached_wrapper,
                            enumerate(indirect_jobs),
                            debug=debug,
                            verbose=verbose,
                            show_progress=show_progress)
            if verbose: print('Passing function as source code worked.')
            return result
        except (SystemExit,KeyboardInterrupt): raise
        except:
            if verbose: 
                traceback.print_exc()
                print('Recompiling source failed, trying to pass as global')
    ############################################################################
    # Try to store function in global dictionary then reset workers.
    if allow_fallback:
        try:
            '''
            Fallback solution by storing function in global scope, then restarting
            the worker threads (who then should be able to see the function) 
            '''
            reset_pool()
            result = __parimap_builtin(f,jobs,
                            debug=debug,
                            verbose=verbose,
                            show_progress=show_progress)
            if verbose: print('Resetting worker-pool worked.')
            return result
        except (SystemExit,KeyboardInterrupt): raise
        except PicklingError:
            if verbose:
                print('Function could not be pickled.')
        except:
            if verbose: 
                traceback.print_exc()
                print('Resetting workers failed.')
        ########################################################################
        # If the above failed, we can try one more thing: pass the function 
        # indirectly by its key in __indirect_eval_fallback_cache dictionary
        try:
            fingerprint = function_fingerprint(f)
            __indirect_eval_fallback_cache[key]=f,None
            reset_pool()
            eval_from_cached(fingerprint,jobs[0],verbose)
            parallel_cached_wrapper((0,indirect_jobs[0]))
            result = parmap(parallel_cached_wrapper,enumerate(indirect_jobs),
                            debug=debug,
                            verbose=verbose,
                            show_progress=show_progress)
            if verbose: print('Reset + pass-source worked')
            return result
        except (SystemExit,KeyboardInterrupt): raise
        except:
            if verbose: 
                traceback.print_exc()
                print('All approaches failed')
    # Everything failed
    raise ValueError('Unable to execute parallel map')

def close_pool(context=None,verbose=False):
    '''
    Safely halts the worker-pool. If worker threads are stuck, 
    then this function will hang. On the other hand, it avoids doing 
    anything violent to close workers. 
    
    Other Parameters
    ----------------
    leavefree : `int`, default 1
        How many cores to "leave free"; The pool size will be the number of
        system cores minus this value
    context : python context, default None
        This context will be used for all workers in the pool
    verbose : `bool`, default False
        Whether to print logging information.
    '''
    global mypool, reference_globals

    # try to see what the calling function sees
    if not context is None:
        reference_globals = context

    if not 'mypool' in globals() or mypool is None:
        if verbose:
            print('No pool found')
    else:
        if verbose:
            print('Pool found, restarting')
            print('Attempting to terminate pool, may become unresponsive')
        # http://stackoverflow.com/questions/16401031/python-multiprocessing-pool-terminate
        def _close_pool():
            global mypool
            if not 'mypool' in globals() or mypool is None:
                return
            sys.stderr.write('\nClosing...')
            mypool.close()
            sys.stderr.write('\n(ok) Terminating...')
            mypool.terminate()
            sys.stderr.write('\n(ok) Joining...')
            mypool.join()
            sys.stderr.write('\n(ok)...')
        def term(*args,**kwargs):
            sys.stderr.write('\nStopping.')
            stoppool=threading.Thread(target=_close_pool)
            stoppool.daemon=True
            stoppool.start()
        signal.signal(signal.SIGTERM, term)
        signal.signal(signal.SIGINT,  term)
        signal.signal(signal.SIGQUIT, term)
        del mypool

def reset_pool(leavefree=None,context=None,verbose=False):
    '''
    Safely halts and restarts the worker-pool. If worker threads are stuck, 
    then this function will hang. On the other hand, it avoids doing 
    anything violent to close workers. 
    
    Other Parameters
    ----------------
    leavefree : `int`, default 1
        How many cores to "leave free"; The pool size will be the number of
        system cores minus this value
    context : python context, default None
        This context will be used for all workers in the pool
    verbose : `bool`, default False
        Whether to print logging information.
    '''
    global mypool, reference_globals
    close_pool(context,verbose)
    NCPU = cpu_count()
    if leavefree is None:
        leavefree = NCPU//2     
    mypool = Pool(max(1,NCPU-leavefree))

def parallel_error_handling(f):
    '''
    We can't really use exception handling in parallel calls.
    This is a wrapper to just catch errors to prevent them from
    propagating up.

    Parameters
    ----------
    
    Returns
    -------
    '''
    def parallel_helper(args):
        try:
            return f(*args)
        except Exception as exc:
            traceback.print_exc()
            info = traceback.format_exc()
            return RuntimeError(info, exc)
    return parallel_helper
