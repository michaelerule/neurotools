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
from itertools import izip, chain
import traceback
import sys
import signal

__N_CPU__ = cpu_count()
'''
Issue with terminate failing. Could be linux python issue. These 
stackoverflow lines may be handy later

process=Process(target=foo,args=('bar',))
pid=process.pid
process.terminate() # works on Windows only

...

os.sytem('kill -9 {}'.format(pid))

'''
# TODO: modify to operate over ND arrays?
def parmap(f,problems,leavefree=1,fakeit=False,verbose=False):
    global mypool
    '''
    f must return an index as the first element in a tuple for this to work
    '''
    if fakeit:
        results = {}
        problems = list(problems)
        num_tasks = len(problems)
        for i,job in enumerate(problems):
            result = f(job)
            sys.stderr.write('\rdone %0.1f%% '%((100.0*(i+1)/num_tasks)))
            results[result[0]] = result[1:]
    else:
        if not 'mypool' in globals() or mypool is None:
            if verbose: print 'NO POOL FOUND. RESTARTING.'
            mypool = Pool(cpu_count()-leavefree)
            return parmap(f,problems)
        else:
            # execute jobs in parallel
            results   = {}
            problems  = list(problems)
            num_tasks = len(problems)
            if verbose: print 'STARTING PARALLEL'
            for i,result in enumerate(mypool.imap_unordered(f,problems)):
                sys.stderr.write('\rdone %0.1f%% '%((100.0*(i+1)/num_tasks)))
                results[result[0]] = result[1:]
            if verbose: print 'FINISHED PARALLEL'
    sys.stderr.write('\r            \r')
    return [results[i] for i in sorted(list(results.keys()))]
    
def parmap_dict(f,problems,leavefree=1,fakeit=False,verbose=False):
    global mypool
    '''
    f must return an index as the first element in a tuple for this to work
    '''
    if fakeit:
        results = {}
        problems = list(problems)
        num_tasks = len(problems)
        for i,job in enumerate(problems):
            result = f(job)
            sys.stderr.write('\rdone %0.1f%% '%((100.0*(i+1)/num_tasks)))
            results[result[0]] = result[1:]
    else:
        if not 'mypool' in globals() or mypool is None:
            if verbose: print 'NO POOL FOUND. RESTARTING.'
            mypool = Pool(cpu_count()-leavefree)
            return parmap(f,problems)
        else:
            # execute jobs in parallel
            results   = {}
            problems  = list(problems)
            num_tasks = len(problems)
            if verbose: print 'STARTING PARALLEL'
            for i,result in enumerate(mypool.imap_unordered(f,problems)):
                sys.stderr.write('\rdone %0.1f%% '%((100.0*(i+1)/num_tasks)))
                results[result[0]] = result[1:]
            if verbose: print 'FINISHED PARALLEL'
    sys.stderr.write('\r            \r')
    return results

def reset_pool(leavefree=1):
    global mypool
    if not 'mypool' in globals() or mypool is None:
        print 'NO POOL FOUND. STARTING'
        mypool = Pool(cpu_count()-leavefree)
    else:
        print 'POOL FOUND. RESTARTING'
        print 'Attempting to terminate pool, may become unresponsive'
        '''
        mypool.terminate()
        mypool.join()
        mypool.close()
        '''
        # might fix the freezing issues
        # http://stackoverflow.com/questions/16401031/python-multiprocessing-pool-terminate
        def close_pool():
            global mypool
            mypool.close()
            mypool.terminate()
            mypool.join()
        def term(*args,**kwargs):
            sys.stderr.write('\nStopping...')
            stoppool=threading.Thread(target=close_pool)
            stoppool.daemon=True
            stoppool.start()
        signal.signal(signal.SIGTERM, term)
        signal.signal(signal.SIGINT, term)
        signal.signal(signal.SIGQUIT, term)
        #
        del mypool
        mypool = Pool(cpu_count()-leavefree)






def parmap_enum(f,problems):
    '''
    f must return an index as the first element in a tuple for this to work
    '''
    mypool = Pool(cpu_count())
    results = {}
    for i,result in enumerate(mypool.imap_unordered(f,problems)):
        sys.stderr.write('\r%d done'%i)
        results[result[0]] = result[1:]
    print 'Attempting to terminate pool, may become unresponsive'
    mypool.terminate()
    mypool.join()
    mypool.close()
    del mypool
    return [results[i] for i in sorted(list(results.keys()))]
