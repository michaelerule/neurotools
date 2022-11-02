#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Code dealing with closures in Python.

The default way that Python handles closures creates problems

 1. Late binding can cause suprising errors
 2. Closing over mutable state can cause surprising errors
 3. Mutable default arguments can causing surprising errors
 4. Closing over external state that may change across runs can cause
    surprising errors, as the same arguments may return different values
 5. Closing over extrenal state makes safe caching difficult, as we
    have to guarantee that the arguments, function source, subroutines,
    and closed-over state are all IDENTICAL before we can trust a
    cached value.

This module defines some decorators that improve safety, hopefully
automatically detecting functions that are likele to cause trouble.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import inspect
import types
import sys
if sys.version_info > (3,): long=int

def is_immutable(x):
    '''
    Checks whether an object is immutable. It must be composed of the
    built-in types str, int, long, bool, float, and tuple. If it is
    a tuple, each of its elements must be immutable
    '''
    if sys.version_info > (3,):
        safe = (str, int, complex, bool, float, type, type(None))
    else:
        safe = (str, int, long, bool, float, type(None))
    if type(x) in safe: return True
    if type(x)==tuple: return len(x)==0 or all([is_immutable(i) for i in x])
    return False

def is_probably_safe(x):
    '''
    Objects are probably "safe" (unlikly to change) if they are
    -- immutable
    -- functions
    -- modules
    Obviously, the latter two may change, but in practice it is likely
    ok. Still, risky!
    '''
    if is_immutable(x): return True
    if sys.version_info > (3,):
        probably_fine = (\
            types.LambdaType,
            types.BuiltinMethodType,
            types.BuiltinFunctionType,
            types.FunctionType,
            types.ModuleType,
            types.MethodType)
    else:
        probably_fine = (\
            types.LambdaType,
            types.InstanceType,
            types.NoneType,
            types.NotImplementedType,
            types.TypeType,
            types.UnicodeType,
            types.ComplexType,
            types.ClassType,
            types.BuiltinMethodType,
            types.BuiltinFunctionType,
            types.FunctionType,
            types.ModuleType,
            types.MethodType)

    if type(x) in probably_fine: return True
    if hasattr(x,'__call__'): return True
    return False

def csv(x):
    return ', '.join(map(str,x))

def inspect_function_closure(f):
    '''
    Checks if a function is "safe"

    Formally this is quite tricky as not all closures can be avoided.
    Functions MUST close over their namespace in some sense, in order
    to access subroutines. This cannot be avoided, and changes in
    subroutines may cause changes in behavior.
    '''

    # Code object can tell us more information about this function
    fc = f.__code__

    # The names attribute of the code object tells use things that are
    # treated like "globals" to this function. These can be variables
    # closed over, or functions and subroutines from the namespace in
    # which the function was defined
    # Bound names can be retrieved from the globals dictionary for
    # this function. Let's make a mini snapshot of that scope
    bound = {k:f.__globals__[k] for k in fc.co_names if k in f.__globals__}
    if len(bound):
        print('Bound variables are')
        for k,v in bound.iteritems(): print('\t %s: \t%s'%(k,v))

    # Free variables are also part of the closed-over state. We really don't
    # want to see these if we need to cache functions. Each of these should
    # match 1:1 with an entry in the closure list.
    free = fc.co_freevars
    print('Free variabes are',csv(free))

    # The closure object lists a collection of cells over which this
    # function closes. These are technically mutable, but often are in
    # practice fixed, or fixed except in exceptional circumstances.
    closure = f.__closure__

    # Sanity check that we have a reference for every free variable.
    # I think this is always true but just in case, double check it.
    if closure is None: assert len(free)==0
    else: assert len(closure) == len(free)

    # Check out the free variables, are they OK? What's going on here?
    if closure and  len(closure)>0:
        print("Warning, function closes over potentially mutable scope.")
        print("This is quite dangerous.")
        for i,(n,c) in enumerate(zip(free,closure)):
            x = c.cell_contents
            print('\tValue of free variable %s is %s'%(n,x))
            if is_probably_safe(x):
                print('\t\tCell contents themselves appear to be immutable, but cell may be mutable.')
                print('\t\tIt may be possible to freeze a snapshot of this state when the function is called, enabling proper caching')
            else:
                print('\t\tMutable cell contents, this is very sketchy!')

    # Make sure all default arguments are immutable
    if f.__defaults__:
        for d in f.__defaults__:
            if not is_probably_safe(d): print('Mutable default')


def verify_function_closure(f):
    '''
    Checks if a function is "safe" (likely immutable)

    Formally this is quite tricky as not all closures can be avoided.
    Functions MUST close over their namespace in order to access
    subroutines later, as Python is by design late-binding. This cannot
    be avoided, and changes in subroutines may cause changes in behavior.

    Levels of safety:

    Safe:
        Function references only other functions. All state is passed
        via arguments. Hash values of subroutines are incorporated into
        the hash value for this function.

    Risky:
        Function may close over state and reference globals, as long as
        those states / globals are themselves immutable. This is risky
        as values may still change across invokation. In theory we could
        snapshot the dependent state and incorporate that into the call
        signature.

    Unsafe:
        Function contains mutable defaults. Function references mutable
        globals or closes over mutable state. Unsafe conditions will
        cause a ValueError
    '''

    # The names attribute of the code object tells use things that are
    # treated like "globals" to this function. These can be variables
    # closed over, or functions and subroutines from the namespace in
    # which the function was defined. Bound names can be retrieved from
    # the globals dictionary for this function.
    fc = f.__code__
    bound = {k:f.__globals__[k] for k in fc.co_names if k in f.__globals__}
    # Make sure closed-over and bound variables are... reasonable
    for k,v in bound.items():
        if not is_probably_safe(v):
            raise ValueError('Function %s is not safe for memoization/caching, it closes over the mutable object %s=%s!'%(f.__name__,k,v))

    # Free variables are also part of the closed-over state. We really don't
    # want to see these if we need to cache functions. Each of these should
    # match 1:1 with an entry in the closure list. The closure object lists
    # a collection of cells over which this function closes. These are
    # technically mutable, but often are in practice fixed, or fixed except
    # in exceptional circumstances.
    if f.__closure__:
        closed = dict(zip(f.__code__.co_freevars,f.__closure__))
        # Make sure closed-over and bound variables are... reasonable
        for k,v in closed.items():
            if not is_probably_safe(v):
                raise ValueError('Function %s is not safe for memoization/caching, it closes over the mutable variable %s %s!'%(f.__name__,k,v))

    # Make sure all default arguments are immutable
    if f.__defaults__:
        for v in f.__defaults__:
            if not is_probably_safe(v):
                raise ValueError('Function %s is not safe for memoization/caching, it uses a mutable default %s!'%(f.__name__,v))

    #print('Function %s seems ok'%f.__name__)
    return True


def get_subroutines(f):
    '''
    Returns things that resemble subroutines of function f. These are all
    callable objets that are either named or bound via closures.
    Return value is a set of functions.
    Non-function callable subroutines will cause an error.
    '''

    raise NotImplementedError("Impossible due to dynamic typing, late binding")

    # The names attribute of the code object tells use things that are
    # treated like "globals" to this function. These can be variables
    # closed over, or functions and subroutines from the namespace in
    # which the function was defined. Bound names can be retrieved from
    # the globals dictionary for this function.
    fc = f.__code__
    bound = {k:f.__globals__[k] for k in fc.co_names if k in f.__globals__}

    # Free variables are also part of the closed-over state. We really don't
    # want to see these if we need to cache functions. Each of these should
    # match 1:1 with an entry in the closure list. The closure object lists
    # a collection of cells over which this function closes. These are
    # technically mutable, but often are in practice fixed, or fixed except
    # in exceptional circumstances.
    closed = dict(zip(f.__code__.co_freevars,f.__closure__))

    # It's possible that some callables are passed as default arguments
    default = f.__defaults__

    # Collect things that resemble subroutines. These are set comprehensions
    # because we don't really care what things are called, so much as which
    # function object they point to.
    bound_subroutines   = {v for k,v in bound.items()  if hasattr(v,'__call__')}
    closed_subroutines  = {v for k,v in closed.items() if hasattr(v,'__call__')}
    default_subroutines = {v for v   in default        if hasattr(v,'__call__')}
    routines = bound_subroutines | closed_subroutines | default_subroutines

    # Currently, we can't handle callables in general. In the case that
    # we have callables that are not functions, we need to raise an
    # exception.
    if not all([type(g) is types.FunctionType for g in routines]):
        raise NotImplementedError("Non-function callables aren't supported")

    return routines


if __name__=="__main__":

    # Construct a complicated test function
    a = 3
    b = [6]
    c = ([5],3)
    def outer_test(d,e=4,f=[]):
        g = 34
        h = [15]
        def inner_test(i,j=4,k=[]):
            # closing over many variables
            # a,b,c,d,e,f,g,h,i,j,k
            return csv((a,b,c,d,e,f,g,h,i,j,k)),os.system
        return inner_test

    # Construct a simple test function
    def simple_test(a,b):
        return dot(a,b)

    f = outer_test('hi')
    g = simple_test

    inspect_function_closure(g)
    inspect_function_closure(f)

    verify_function_closure(g)
    verify_function_closure(f)
