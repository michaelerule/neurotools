#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Robust decorators are provided by the decorator package
    http://pythonhosted.org/decorator/documentation.html
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import sys
__PYTHON_2__ = sys.version_info<(3, 0)

import neurotools.util
from collections import defaultdict
import os, sys
import inspect, ast, types
import warnings, traceback, errno
import pickle, json, base64, zlib
import numpy as np

try:
    import decorator
    #from decorator import decorator as robust_decorator
    #import decorator.decorator as robust_decorator
    robust_decorator = decorator.decorator
    #print(robust_decorator)
    #sys.exit(-3)
except:
    traceback.print_exc()
    print('could not find decorator module; '
        'no robust_decorator support')
    #robust_decorator = lambda x:x
    print('This is important; you should not continue!')
    # Null version; migth not work
    def robust_decorator(caller, _func=None):
        if _func is not None:
            return _func
        def g(*args,**kwargs):
            return _func(*args,**kwargs)
        return g
try:
    import typedecorator
    from typedecorator import params, returns, setup_typecheck
except:
    print('could not find typedecorator module; '
        'advanced decorator functions missing')

def listit(t):
    '''
    Converts nested tuple to nested list
    
    Parameters
    ----------
    t: tuple, list, or object
        If tuple or list, will recursively apply listit
    
    Returns
    -------
    
    '''
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

def tupleit(t):
    '''
    Converts nested list to nested tuple
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    return tuple(map(tupleit, t)) if isinstance(t, (list, tuple)) else t

def sanitize(sig,mode='liberal'):
    '''
    Converts an argument signature into a standard format. 
    Lists will be converted to tuples. Non-hashable types 
    will cause an error.

    "strict" mode requires that all data be numeric 
    primitives or strings containing "very safe" chracters 
    a-zA-Z0-9 and space.
    
    Parameters
    ----------
    sig: nested tuple
        Result of calling `argument_signature()`, 
    
    Returns
    -------
    sig: nested tuple
        
    '''
    SAFE = ('qwertyuiopasdfghjklzxcvbnm'
        'QWERTYUIOPASDFGHJKLZXCVBNM'
        '1234567890 ')
    
    if isinstance(sig,np.ndarray):
        if sig.ndim==0:
            sig = sig[()]
        return sanitize(tuple(sig))
    
    if isinstance(sig,np.core.memmap):
        sig = np.array(sig)
        return sanitize(sig)
    
    if isinstance(sig, (list,tuple)):
        if len(sig)<=0: return ()
        if len(sig)==1: return sanitize(sig[0],mode=mode)
        else: return tuple(sanitize(s,mode=mode) 
            for s in sig)
    
    if __PYTHON_2__ and isinstance(sig,(unicode,)):
        return sanitize(str(sig),mode=mode)
    
    if isinstance(sig, (dict,)):
        keys = sorted(sig.keys())
        vals = (sig[k] for k in keys)
        return sanitize(zip(keys,vals),mode=mode)
    
    if isinstance(sig, (set,)):
        return tuple(sanitize(s,mode=mode) 
            for s in sorted(list(sig)))
    
    if hasattr(sig, '__iter__') and not isinstance(sig,str):
        try:
            return tuple(sanitize(s,mode=mode) 
                for s in list(sig))
        except:
            # Probably a 0 dimensional iterable,
            # not sure how to handle this?
            pass

    if mode=='strict':
        if type(sig) in (int,float,bool,None):
            return sig
        if type(sig) is str:
            if any([c not in SAFE for c in sig]):
                raise ValueError('Strict mode requires all'
                    ' strings consist of letters, numbers,'
                    ' and spaces')
            return sig
        raise ValueError('Strict mode requires int, float,'
            ' bool, or str types')
    return sig

def summarize_function(f):
    '''
    Prints function information, 
    Used for debugging decorators.
    
    Parameters
    ----------
    f: function
    '''
    print(f)
    for k in dir(f):
        print('\t',k,'\t',end='')
        if not k in ['__globals__','func_globals','func_dict','__dict__']:
            print(f.__getattribute__(k))
        else: print()
    code = f.func_code
    print('\tCODE')
    for k in dir(code):
        if not k[:2]=='__': print('\t\t',k,'\t',code.__getattribute__(k))
    named,vargnames,kwargnames,defaults = inspect.getargspec(f)
    print('\t',named,vargnames,kwargnames,defaults)


def argument_signature(function,*args,**kwargs):
    '''
    Convert the function arguments and values to a unique 
    set. Throws ValueError if the provided arguments cannot 
    match argspec.
    
    Parameters
    ----------
    function: function
        Function fo create signature for
    *args: iterable
        Arguments for `function`
    **kwwargs: dict
        Keyword arguments for `function`
    
    Returns
    -------
    sig: tuple
    '''
    named_store = {} # map from parameter names to values
    named,vargname,kwargname,defaults = inspect.getargspec(
        function)
    # Pattern matching can give rise to lists in the
    # "named" variable returned here. We need to convert
    # these to something hashable.
    named     = sanitize(named)
    available = list(zip(named,args))
    nargs     = 1 if type(available) is str \
        else 0 if available is None else len(available)
    ndefault  = 1 if type(defaults)  is str \
        else 0 if defaults  is None else len(defaults)
    nnamed    = 1 if type(named)     is str \
        else 0 if named     is None else len(named)
    
    # All positional arguments must be filled
    nmandatory = nnamed - ndefault
    if nargs<nmandatory: 
        details = "%s %s %s %s %s %s %s"%(
            available,
            nargs,
            nnamed,
            named,
            ndefault,
            defaults,
            nmandatory)
        raise ValueError(
            'Not enough positional arguments\n'+details)
        
    # Assign available positional arguments to names
    for k,v in available:
        if k in named_store: raise ValueError(
            'Duplicate argument',k)
        named_store[k] = v
        
    # If not all arguments are provided,
    # check **kwargs and defaults
    ndefaulted   = max(0,nnamed - nargs)
    default_map = dict(zip(named[-ndefault:],defaults)) \
        if ndefault>0 else {}
    if ndefaulted>0:
        for k in named[-ndefaulted:]:
            if k in named_store: raise ValueError(
                'Duplicate argument',k)
            named_store[k] = kwargs[k] 
                if k in kwargs else default_map[k]
            if k in kwargs: del kwargs[k]
            
    # Store excess positional arguments in *varargs
    vargs = None
    if len(args)>nnamed:
        if vargname is None:
            raise ValueError(
                'Excess positional arguments, but function'
                ' does not accept *vargs!')
        vargs = args[nnamed:]
        
    # Store excess keyword arguments if fn takes **kwargs
    if len(kwargs):
        if kwargname is None: raise ValueError(
            'Excess keyword arguments, '
            'but function does not accept **kwargs!')
        for k in kwargs:
            if k in named_store: raise ValueError(
                'Duplicate argument',k)
            named_store[k] = kwargs[k]
            
    # Construct a tuple reflecting argument signature
    keys  = sorted(named_store.keys())
    vals  = tuple(named_store[k] for k in keys)
    args  = sanitize(tuple(zip(keys,vals)))
    if len(keys)==1:
        args = (args,)
    result = (args,sanitize(vargs))
    return result

def print_signature(sig):
    '''
    Formats the argument signature for printing.
    
    Parameters
    ----------
    sig: tuple
        Tuple returned by `argument_signature()`
    
    Returns
    -------
    :str
    '''
    named, vargs = sig
    result = ','.join(['%s=%s'%(k,v) for (k,v) in named])
    if not vargs is None: result += ','+','.join(map(str,vargs))
    return result

@robust_decorator
def timed(f,*args,**kwargs):
    '''
    Timer decorator: Modifies a function to reutrn 
    a tuple of (runtime, result).
    
    Parameters
    ----------
    f: function
    *args: iterable
        Arguments for `f`
    **kwargs: dict
        Keyword arguments for `f`
    
    Returns
    -------
    time_taken: int
        Time taken for the function call, in milliseconds
    result:
        Return value of `f(*args,**kwargs)`
    '''
    t0     = neurotools.util.time.current_milli_time()
    result = f(*args,**kwargs)
    t1     = neurotools.util.time.current_milli_time()
    return float(t1-t0), result

__memoization_caches__ = dict()

def clear_memoized(verbose=False):
    '''
    Clear the caches of all memoized functions
    
    Other Parameters
    ----------------
    verbose: boolean; default False
        Print extra debugging information? 
    '''
    global __memoization_caches__
    if verbose:
        print('Purging all memoization caches')
    for f,cache in __memoization_caches__.items():
        cache.clear()

def memoize(f):
    '''
    Memoization decorator
    
    Parameters
    ----------
    f: function
    
    Returns
    -------
    :function
        Memoize-decorated function
    '''
    global __memoization_caches__
    cache = {}
    info  = defaultdict(dict)
    @robust_decorator
    def wrapped(f,*args,**kwargs):
        sig = argument_signature(f,*args,**kwargs)
        if not sig in cache:
            time,result = f(*args,**kwargs)
            info[sig]['density'] = time/sys.getsizeof(result)
            cache[sig] = result
        return cache[sig]
    wrapped.__cache__ = cache
    wrapped.__info__  = info
    result = wrapped(timed(f))
    __memoization_caches__[result] = cache
    return result

def unwrap(f):
    '''
    Strips decorators from a decorated function, provided that the
    decorators were so kind as to set the .__wrapped__ attribute
    
    Parameters
    ----------
    f: function
        Decorated function to unpack
    
    Returns
    -------
    g: function
        Base function obtained by recursively looking for
        the attribute `__wrapped__` in function `f`, which
        stores the original (undecorated) function for
        functions modified by the decorators in 
        `ndecorator`.
    '''
    g = f
    while hasattr(g, '__wrapped__'):
        g = g.__wrapped__
    return g

if __name__=="__main__":

    print("Testing the memoize decorator")

    def example_function(a,b,c=1,d=('ok',),*vargs,**kw):
        ''' This docstring should be preserved by the decorator '''
        e,f = vargs if (len(vargs)==2) else (None,None)
        g = kw['k'] if 'k' in kw else None
        print(a,b,c,d,e,f,g)

    f = example_function
    g = memoize(example_function)

    print('Testing example_function')
    fn = f
    fn('a','b','c','d')
    fn('a','b','c','d','e','f')
    fn('a','b',c='c',d='d')
    fn('a','b',**{'c':'c','d':'d'})
    fn('a','b',*['c','d'])
    fn('a','b',d='d',*['c'])
    fn('a','b',*['c'],**{'d':'d'})
    fn('a','b','c','d','e','f')

    print('Testing memoized example_function')
    fn = g
    fn('a','b','c','d')
    fn('a','b','c','d','e','f')
    fn('a','b',c='c',d='d')
    fn('a','b',**{'c':'c','d':'d'})
    fn('a','b',*['c','d'])
    fn('a','b',d='d',*['c'])
    fn('a','b',*['c'],**{'d':'d'})
    fn('a','b','c','d','e','f')
