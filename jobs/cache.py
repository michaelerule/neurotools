#!/usr/bin/python
# -*- coding: UTF-8 -*-
# BEGIN PYTHON 2/3 COMPATIBILITY BOILERPLATE
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *
import sys
__PYTHON_2__ = sys.version_info<(3, 0)
# END PYTHON 2/3 COMPATIBILITY BOILERPLATE

from   collections import defaultdict
import numpy as np
import scipy.io
import inspect
import ast
import types
import os
import time
import sys
import subprocess
import warnings
import traceback
import errno
import pickle
import json
import base64
import zlib

#something weird here. which pickle does multiprocessing use?
try:
    from cPickle import PicklingError
except:
    from pickle import PicklingError

import neurotools.jobs.decorator
import neurotools.tools
import neurotools.ntime

from neurotools.jobs.closure import verify_function_closure

CACHE_IDENTIFIER ='.__neurotools_cache__'

VERBOSE_CACHING = 0#True

def validate_argument_signature(sig):
    '''
    Determines whether a given argument signature can be used to cache
    things to disk. The argument signature must be hashable. It must
    consists of types that can be translated between python, numpy, and
    matlab convenctions.
    '''
    pass

def is_dangerous_filename(filename):
    if len(filename)>255:
        return True
    if __PYTHON_2__ and type(filename) is unicode:
        return True
    if any([c in filename for c in "\/<>:\"'|?*,@#{}'&`!%$\n\t\b\r "]):
        return True
    return False

def check_filename(filename):
    if len(filename)>255:
        warnings.warn('FILE NAME MAY BE TOO LONG ON SOME SYSTEMS')
    if __PYTHON_2__ and type(filename) is unicode:
        warnings.warn('FILE NAME IS UNICODE')
    if any([c in filename for c in "/?<>\:*|\"\n\t\b\r"]):
        raise ValueError('Filename contains character forbidden on windows')
    if any([c in filename for c in "\/<>:\"'|?*,@#{}'&`!%$\n\t\b\r "]):
        warnings.warn('FILE NAME CONTAINS CHARACTER THAT MAY CAUSE ISSUES IN SOME SOFTWARE')



@neurotools.jobs.decorator.memoize
def function_hash_with_subroutines(f):
    '''
    Functions may change if their subroutines change. This function computes
    a hash value that is sensitive to changes in the source code, docstring,
    argument specification, name, module, and subroutines.

    This is a recursive procedure with a fair amount of overhead.
    To allow for the possibility of mutual recursion, subroutines are
    excluded from the hash if the function has already been visited.

    This does not use the built-in hash function for functions in python.

    Is memoization possible? Making memoization compatible with graceful
    handling of potentially complex mutually recurrent call structures is
    tricky. Each function generates a call tree, which does not expand a
    node if it is already present in the call tree structure. Therefore
    there are many possible hash values for an intermediate function
    depending on how far it's call tree gets expanded, which depends on
    what has been expanded and encountered so far. Therefore, we cannot
    cache these intermediate values.

    Is it worth noting that the topology of a mutually recurrent call
    structure cannot change without changing the source code of at least
    one function in the call graph? So it suffices, to hash the subroutines,
    to expand the call graph (potentially excluding standard and system
    library functions), grab the non-recursive hash for each of these
    functions (which may be cached), and then generate the subroutine
    dependent hash by combining the non-recursive hash with the hash
    of a datastructure representing the subroutine "profile" obtained
    from the call graph.

    For now we are assuming that any decorators wrapping the function
    do not modify it's computation, and therefore can safely be stripped.
    This is an assumption and is not, in general, true.

    Note that this function cannot detect changes in effective function
    behavior that result from changes in global variables or mutable scope
    that has been closed over.
    '''
    raise NotImplementedError("This is actually impossible due to dynamic typing and late binding")

    # repeatedly expand list of subroutines
    to_expand = {f}
    expanded  = set()
    while len(to_expand)>0:
        new_subroutines = set()
        for g in to_expand: new_subroutines |= get_subroutines(g)
        expanded |= to_expand
        to_expand = new_subroutines - expanded
    # we now have a set, we need to provide some ordering over that set
    # sort the hash values and hash that
    return hash(tuple(sorted(map(function_hash_no_subroutines,expanded))))


def get_source(f):
    try:
        source    = inspect.getsource(neurotools.jobs.decorator.unwrap(f))
    except IOError as ioerr:
        if ioerr.message!="could not get source code": raise
        # some dynamically created functions may not have source code
        # to bypass this, we can see if the routine that created the
        # function was kind enough to store the source code for us
        source = neurotools.jobs.decorator.unwrap(f).__source__
    return source

@neurotools.jobs.decorator.memoize
def function_hash_no_subroutines(f):
    '''
    See function_hash_with_subroutines. This has value is based on the

        1   Undecorated source code
        2   Docstring
        3   function name
        4   module name
        5   function argument specification

    Note that this function cannot detect changes in effective function
    behavior as a result of changes in subroutines, global variables, or
    mutable scope that has been closed over.
    '''
    source    = get_source(f)
    docstring = inspect.getdoc(f)
    name      = f.__name__
    module    = f.__module__
    argspec   = neurotools.jobs.decorator.sanitize(inspect.getargspec(f))
    return hash((module,name,docstring,source,argspec,subroutines))

def function_signature(f):
    '''
    Generates identifier used to locate cache corresponding to a
    particular function.

    We want to be able to cache results to dist to memoize across
    different instances and over time. However, if the code for the
    underlying function changes, we're in a pickle, as checking whether
    the change is meaningful is almost impossible.

    Caches can also become invalid if the behavior of subroutines change,
    quite tricky!

    For now, we'll check that the function module, name, argspec, source,
    and file are the same. Note that module and name identify which cache,
    and source, file, and argspec validate that the function has not
    changes significantly.
    '''
    # The one thing the decorator module can't fake is where the
    # function is defined. So we can't see the source code directly if
    # we're passed a wrapped function. We can however detect this case
    # and peel away the layers to get to the underlying source. The
    # decorator module will leave the wrapped function in a variable
    # called __wrapped__, so we can follow this back to the source code
    g = f
    source    = get_source(f)
    docstring = inspect.getdoc(f)
    name      = f.__name__
    module    = f.__module__
    argspec   = neurotools.jobs.decorator.sanitize(inspect.getargspec(f))

    identity  = (module,name)
    signature = (docstring,source,argspec)
    name = '.'.join(identity)
    code = base64.urlsafe_b64encode(\
        str(hash((identity,signature))&0xffff).encode('UTF-8')).decode().replace('=','')
    return name+'.'+code

def signature_to_file_string(f,sig,
    mode='repr',
    compressed=True,
    base64encode=True,
    truncate=True):
    '''
    Converts an argument signature to a string if possible. 
    
    This can
    be used to store cached results in a human-readable format.
    Alternatively, we may want to simply encode the value of the
    argument signature in a string that is compatible with most file
    systems. We'd still need to perform verification on the object.

    No more than 4096 characters in path string
    No more than 255 characters in file string
    For windows compatibility try to limit it to 260 character total pathlen

    For compatibility, these characters should be avoided in paths:
        `\/<>:"|?*,@#={}'&`!%$. ASCII 0..31`

    The easiest way to avoid problematic characters without restricting the
    input is to re-encode as base 64.

    The following modes are supported.

        repr:
            Uses repr and ast.literal_eval(node_or_string) to serialize the
            argument signature. This is save, but restricts the types permitted
            as paramteters.

        json:
            Uses json to serialize the argument signature. Argument signatures
            cannot be uniquely recovered, because tuples and lists both map to
            lists in the json representation. Restricting the types used in
            the argument signature may circumvent this.

        pickle:
            Uses pickle to serialize argument signature. This should uniquely
            store argument signatures that can be recovered, but takes more
            space.

        human:
            Attempts a human-readable format. Eperimental.

    Compression is on by defaut
    Signatures are base64 encoded by default
    '''
    sig = neurotools.jobs.decorator.sanitize(sig)

    if compressed and not base64encode:
        raise ValueError('If you want compression, turn on base64 encoding')

    # A hash value gives us good distribution to control the complexity of
    # the directory tree used to manage the cache, but is not unique
    hsh = base64.urlsafe_b64encode(str(hash(sig)&0xffff).encode('UTF-8')).decode().replace('=','')

    # We also need to store some information about which function this
    # is for. We'll get a human readable name identifying the funciton,
    # and a shorter hash-value to make sure we invalidate the cache if
    # the source code or function definition changes.
    fname = function_signature(f)

    # The argument spec can be mapped uniquely to a file name by converting
    # it to text, then converting this text to base64 to avoid issues with
    # special characters. Passing the text representation through zlib
    # preserves the uniqueness of the key, while reducing the overall size.
    # This improves performance
    # convert key to an encoded string
    if   mode=='repr'  : key = repr(sig)
    elif mode=='json'  : key = json.dumps(sig)
    elif mode=='pickle': key = pickle.dumps(sig)
    elif mode=='human' : key = human_encode(sig)
    else: raise ValueError('I support coding modes repr, json, and pickle\n'+
        'I do not recognize coding mode %s'%mode)
    # compress and base64 encode string
    key = key.encode('UTF-8')
    if compressed  : key = zlib.compress(key)
    if base64encode: key = base64.urlsafe_b64encode(key)

    # Path will be a joining of the hash and the key. The hash should give
    # good distribution, while the key means we can recover the arguments
    # from the file name.
    filename = '%s.%s.%s'%(fname,hsh,key.decode())
    # If for some reason the path is too long, complain
    if len(filename)>255:
        if truncate:
            # hash the key if it is too long and truncation is enabled
            # TODO: probably should be a better hash function?
            s  = key.decode()
            kh = base64.urlsafe_b64encode(str(hash(s)).encode('UTF-8')).decode().replace('=','')
            filename = '%s.%s.%s'%(fname,hsh,kh)
            filename = filename[:255]
        else:
            raise ValueError(\
                'Argument specification exceeds maximum path length.\n'+
                'Function probably accepts data as an argument,\n'+
                'rather than a key to locate data. See Joblib for a\n'+
                'caching framework that uses cryptographic hashes\n'+
                'to solve this problem. For now, we skip the cache.\n\n'+
                'The offending filename is '+filename)
    check_filename(filename)
    return filename

def file_string_to_signature(filename,mode='repr',compressed=True,base64encode=True):
    '''
    Extracts the argument key from the compressed representation in a
    cache filename entry. Inverse of signature_to_file_string.

    The following modes are supported.

    repr:
        Uses repr and ast.literal_eval(node_or_string) to serialize the
        argument signature. This is safe, but restricts the types permitted
        as paramteters.

    json:
        Uses json to serialize the argument signature. Argument signatures
        cannot be uniquely recovered, because tuples and lists both map to
        lists in the json representation. Restricting the types used in
        the argument signature may circumvent this.

    pickle:
        Uses pickle to serialize argument signature. This should uniquely
        store argument signatures that can be recovered, but takes more
        space.

    human:
        Attempts a human-readable format. Eperimental.

    Compression is on by default
    Signatures are base64 encoded by default
    '''
    pieces = filename.split('.')
    key  = pieces[-1]
    hsh  = pieces[-2]
    name = '.'.join(pieces[:-3])

    # The argument spec can be mapped uniquely to a file name by converting
    # it to text, then converting this text to base64 to avoid issues with
    # special characters. Passing the text representation through zlib
    # preserves the uniqueness of the key, while reducing the overall size.
    # This improves performance
    if base64encode: key = base64.urlsafe_b64decode(key.encode('UTF-8'))
    if compressed  : key = zlib.decompress(key)
    key = key.decode()
    if   mode=='repr'  : sig = ast.literal_eval(key)
    elif mode=='json'  : sig = json.loads(key)
    elif mode=='pickle': sig = pickle.loads(key)
    elif mode=='human' : sig = human_decode(key)
    else: raise ValueError('I support coding modes repr, json, and pickle\n'+
        'I do not recognize coding mode %s'%mode)
    sig = neurotools.sanitize(sig)
    return sig

def human_encode(sig):
    '''Formats the argument signature for saving as file name'''
    sig = neurotools.sanitize(sig,mode='strict')
    named, vargs = sig
    if not vargs is None:
        raise ValueError(
            'Currently variable arguments are not permitted '+
            'in the human-readable format')
    result = ','.join(['%s=%s'%(k,repr(v)) for (k,v) in named])
    return result

def human_decode(key):
    '''Formats the argument signature for saving as file name'''
    params = [k.split('=') for k in key.split(',')]
    params = tuple((n,ast.literal_eval(v)) for n,v in params)
    sig = (params,None)
    sig = neurotools.sanitize(sig,mode='strict')
    return sig

def locate_cached(cache_root,f,method,*args,**kwargs):
    sig = neurotools.jobs.decorator.argument_signature(f,*args,**kwargs)
    fn  = signature_to_file_string(f,sig,
            mode        ='repr',
            compressed  =True,
            base64encode=True)

    pieces   = fn.split('.')
    path     = cache_root + os.sep + os.sep.join(pieces[:-2]) + os.sep
    filename = '.'.join(pieces[-2:])+'.'+method
    location = path+filename
    return fn,sig,path,filename,location

def validate_for_matfile(x):
    '''
    Numpy types: these should be compatible
    ==========  ================================================================================
    Type        Description
    ==========  ================================================================================
    bool\_ 	    Boolean (True or False) stored as a byte
    int8 	    Byte (-128 to 127)
    int16 	    Integer (-32768 to 32767)
    int32 	    Integer (-2147483648 to 2147483647)
    int64 	    Integer (-9223372036854775808 to 9223372036854775807)
    uint8 	    Unsigned integer (0 to 255)
    uint16 	    Unsigned integer (0 to 65535)
    uint32 	    Unsigned integer (0 to 4294967295)
    uint64 	    Unsigned integer (0 to 18446744073709551615)
    float16 	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    float32 	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    float64 	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    complex64 	Complex number, represented by two 32-bit floats (real and imaginary components)
    complex128 	Complex number, represented by two 64-bit floats (real and imaginary components)
    ==========  ================================================================================
    '''
    safe = (np.bool_  , np.int8     , np.int16 , np.int32 , np.int64  ,
                  np.uint8  , np.uint16   , np.uint32, np.uint64, np.float32,
                  np.float64, np.complex64, np.complex128)
    if not type(x) == np.ndarray: x = np.array(x)
    if len(shape(x))<2:
        raise ValueError("One-dimensional arrays cannot be stored safely in matfiles")
    if x.dtype == np.object:
        # object arrays will be converted to cell arrays,
        # we need to make sure each cell can be stored safely
        return map(validate_for_matfile,x)
    if not x.dtype in safe:
        raise ValueError("Numpy type %s is not on the list of compatible types"%x.dtype)
    return True


def validate_for_numpy(x):
    '''
    Numpy types: these should be compatible
    ==========  ================================================================================
    Type        Description
    ==========  ================================================================================
    bool\_ 	    Boolean (True or False) stored as a byte
    int8 	    Byte (-128 to 127)
    int16 	    Integer (-32768 to 32767)
    int32 	    Integer (-2147483648 to 2147483647)
    int64 	    Integer (-9223372036854775808 to 9223372036854775807)
    uint8 	    Unsigned integer (0 to 255)
    uint16 	    Unsigned integer (0 to 65535)
    uint32 	    Unsigned integer (0 to 4294967295)
    uint64 	    Unsigned integer (0 to 18446744073709551615)
    float16 	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    float32 	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    float64 	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    complex64 	Complex number, represented by two 32-bit floats (real and imaginary components)
    complex128 	Complex number, represented by two 64-bit floats (real and imaginary components)
    ==========  ================================================================================
    '''
    safe = (np.bool_  , np.int8     , np.int16 , np.int32 , np.int64  ,
                  np.uint8  , np.uint16   , np.uint32, np.uint64, np.float32,
                  np.float64, np.complex64, np.complex128)
    if not type(x) == np.ndarray:
        x = np.array(x)
    if x.dtype == np.object:
        # object arrays will be converted to cell arrays,
        # we need to make sure each cell can be stored safely
        try:
            ix = iter(x)
        except TypeError as te:
            raise ValueError('is not iterable')
        return map(validate_for_numpy,x)
    if not x.dtype in safe:
        raise ValueError("Numpy type %s is not on the list of compatible types"%x.dtype)
    return True

def disk_cacher(
    cache_location,
    method='npy',
    write_back=True,
    skip_fast=False,
    verbose=VERBOSE_CACHING,
    allow_mutable_bindings=False):
    '''
    Decorator to memoize functions to disk.
    Currying pattern here where cache_location creates decotrators

    write_back:

         True: Default. Computed results are saved to disk

        False: Computed results are not saved to disk. In this case of
               Hierarchical caches mapped to the filesystem, a background
               rsync loop can handle asynchronous write-back.

    method:

         p: Use pickle to store cache. Can serialize all objects but
            seriously slow! May not get ANY speedup due to time costs if
            pickling and disk IO

       mat: Use scipy.io.savemat and scipy.io.loadmat. Nice because it's
            compatible with matlab. Unfortunately, can only store numpy types
            and data that can be converted to numpy types. Data conversion
            may alter the type signature of the return arguments when
            retrieved from the cache.

       npy: Use built in numpy.save functionality. Experimental. Will
            likely only work if the return value is a single numpy array?

      hdf5: Not supported. Will be implemented in the future
    '''
    VALID_METHODS = ('pickle','mat','npy')
    assert method in VALID_METHODS
    cache_location = os.path.abspath(cache_location)+os.sep
    cache_root     = cache_location+CACHE_IDENTIFIER
    def cached(f):
        if not allow_mutable_bindings:
            verify_function_closure(f)
        @neurotools.jobs.decorator.robust_decorator
        def wrapped(f,*args,**kwargs):
            t0 = neurotools.ntime.current_milli_time()
            try:
                fn,sig,path,filename,location = locate_cached(cache_root,f,method,*args,**kwargs)
            except ValueError as exc:
                print('Generating cache key has failed')
                print('Skipping chaching entirely')
                try:
                    traceback.print_exc(exc)
                except:
                    print('Error occurred while printing traceback')
                    print(repr(exc))
                time,result = f(*args,**kwargs)
                return result
            try:
                if method=='pickle':
                    result = pickle.load(open(location,'rb'))
                elif method =='mat':
                    result = scipy.io.loadmat(location)['varargout']
                elif method =='npy':
                    try:
                        result = np.load(location, mmap_mode='r')
                    except ValueError:
                        result = np.load(location)
                if verbose:
                    print('Retrieved cache at ',path)
                    print('\t%s.%s'%(f.__module__,f.__name__))
                    print('\t%s'%neurotools.jobs.decorator.print_signature(sig))
            except (EOFError, IOError) as exc:
                if verbose:
                    print('Recomputing cache at %s'%cache_location)
                    print('\t%s.%s'%(f.__module__,f.__name__))
                    print('\t%s'%neurotools.jobs.decorator.print_signature(sig))

                time,result = f(*args,**kwargs)
                neurotools.tools.ensure_dir(path)

                if verbose: print('\tTook %d milliseconds'%time)

                if write_back:
                    if verbose: print('Writing cache at ',path)

                    try:
                        if method=='pickle':
                            pickle.dump(result,
                                open(location,'wb'),
                                protocol=pickle.HIGHEST_PROTOCOL)
                        elif method =='mat':
                            validated_result = validate_for_matfile(result)
                            if validated_result is None:
                                raise ValueError('Error: return value cannot be safely packaged in a matfile')
                            scipy.io.savemat(location,{
                                'varargout':result})
                        elif method =='npy':
                            validated_result = validate_for_numpy(result)
                            if validated_result is None:
                                raise ValueError('Error: return value cannot be safely packaged in a numpy file')
                            np.save(location, result)
                    except (RuntimeError, ValueError, IOError, PicklingError) as exc2:
                        if verbose:
                            print('Saving cache at %s FAILED'%cache_location)
                            print('\t%s.%s'%(f.__module__,f.__name__))
                            print('\t%s'%\
                                neurotools.jobs.decorator.print_signature(sig))
                            print('\n\t'.join(\
                                traceback.format_exc().split('\n')))

                    if verbose:
                        try:
                            print('Wrote cache at ',path)
                            print('\tFor function %s.%s'%\
                                (f.__module__,f.__name__))
                            print('\tArgument signature %s'%\
                                neurotools.jobs.decorator.print_signature(sig))
                            st        = os.stat(location)
                            du        = st.st_blocks * st.st_blksize
                            t1        = neurotools.ntime.current_milli_time()
                            overhead  = float(t1-t0) - time
                            io        = float(du)/(1+overhead)
                            recompute = float(du)/(1+time)
                            boost     = (recompute-io)
                            saved     = time - overhead
                            quality   = boost/(1+float(du))
                            print('\tSize on disk is %d'%du)
                            print('\tIO overhead %d milliseconds'%overhead)
                            print('\tCached performance %0.4f'%io)
                            print('\tRecompute cost     %0.4f'%recompute)
                            print('\tExpected boost     %0.4f'%boost)
                            print('\tTime-space quality %0.4f'%quality)
                        except (OSError) as exc3:
                            print('\n\t'.join(\
                                traceback.format_exc().split('\n')))
                    if skip_fast and boost<0:
                        if verbose:
                            print('\tWARNING DISK IO MORE EXPENSIVE THAN RECOMPUTING!')
                            print('\tWe should really do something about this?')
                            print('\tZeroing out the file, hopefully that causes it to crash on load?')
                        with open(location, 'w'): pass
            return result
        def purge(*args,**kwargs):
            '''
            Delete cache entries matching arguments.
            '''
            for method in VALID_METHODS:
                fn,sig,path,filename,location = locate_cached(cache_root,f,method,*args,**kwargs)
                print('Deleting %s'%location)
                try:
                    os.remove(location)
                    print('Deleted %s'%location)
                except OSError as ee:
                    if ee.errno==2:
                        print('%s does not exist'%location)
                    else:
                        raise
            pass
        decorated = wrapped(neurotools.jobs.decorator.timed(f))
        decorated.purge = purge
        return decorated
    return cached


def hierarchical_cacher(
        fast_to_slow,
        method='npy',
        write_back=True,
        verbose=VERBOSE_CACHING,
        allow_mutable_bindings=False):
    '''
    Designed for constructing a hierarchy of disk caches.
    '''
    slow_to_fast = fast_to_slow[::-1] # reverse it
    all_cachers  = []
    def hierarchical(f):
        # disable write-back on the slow caches
        for location in slow_to_fast[:-1]:
            f = disk_cacher(location,method=method,write_back=write_back,verbose=verbose,allow_mutable_bindings=allow_mutable_bindings)(f)
            all_cachers.append(f)
        # use write-back only on the fast cache
        location = slow_to_fast[-1]
        f = disk_cacher(location,method=method,write_back=True,verbose=verbose,allow_mutable_bindings=allow_mutable_bindings)(f)
        def purge(*args,**kwargs):
            '''
            Purge each of the constituent cachers
            '''
            for cacher in all_cachers:
                if hasattr(cacher,'purge'):
                    cacher.purge(*args,**kwargs)
        f.purge = purge
        return f
    return hierarchical





######################################################################
# Setup advanced memoization

import os
myhost = os.uname()[1]
if myhost in ('moonbase',):
    ramdisk_location   = '/media/neurotools_ramdisk'
    ssd_cache_location = '/ssd_1/mrule'
    hdd_cache_location = '/ldisk_1/mrule'
elif myhost in ('basecamp',):
    ramdisk_location   = '/media/neurotools_ramdisk'
    ssd_cache_location = '/home/mrule'
elif myhost in ('RobotFortress','petra'):
    ramdisk_location   = '/Users/mrule/neurotools_ramdisk'
    ssd_cache_location = '/Users/mrule'
else:
    print('New System. Cache Locations will need configuring.')
    print('TODO: clean this up somehow! hard-coded configurations are not acceptable in public code')
    ramdisk_location = ssd_cache_location = hdd_cache_location = None


disk_cache_hierarchy = (
    ramdisk_location,
    ssd_cache_location)

def purge_ram_cache():
    os.system('rm -rf ' + ramdisk_location + os.sep + CACHE_IDENTIFIER)

def purge_ssd_cache():
    os.system('rm -rf ' + ssd_cache_location + os.sep + CACHE_IDENTIFIER)

def du(location):
    st = os.stat(location)
    du = st.st_blocks * st.st_blksize
    return du

def reset_ramdisk():
    print('Initializing public ramdisk at %s'%ramdisk_location)
    os.system('sudo mkdir -p '+ramdisk_location)
    os.system('sudo umount %s/'%ramdisk_location)
    os.system('sudo mount -t tmpfs -o size=500G tmpfs %s/'%ramdisk_location)
    os.system('sudo chmod -R 777 %s'%ramdisk_location)
if not os.path.isdir(ramdisk_location): reset_ramdisk()

disk_cached       = disk_cacher('.')
leviathan         = hierarchical_cacher(disk_cache_hierarchy,method='npy')
unsafe_disk_cache = hierarchical_cacher(disk_cache_hierarchy,method='npy',allow_mutable_bindings=True)
pickle_cache      = hierarchical_cacher(disk_cache_hierarchy,method='pickle')

old_memoize = neurotools.jobs.decorator.memoize
new_memoize = leviathan
memoize     = new_memoize
neurotools.jobs.decorator.memoize = new_memoize

def launch_cache_synchronizers():
    assert(0)
    # write_back is disabled for the slow disk caches. To make these
    # persistant, we need to run an rsync job to keep them synchronized.
    disk_cache_hierarchy = (ramdisk_location,ssd_cache_location,hdd_cache_location)
    for level in range(len(disk_cache_hierarchy)-1):
        source       = disk_cache_hierarchy[level  ] + os.sep + CACHE_IDENTIFIER
        destination  = disk_cache_hierarchy[level+1] + os.sep + CACHE_IDENTIFIER
        destination + CACHE_IDENTIFIER
        # quiet rsync command in update-archive mode
        rsync = "rsync -aqu '%s/' '%s' "%(source,destination)
        # Run the synchronization jobs at idle level
        nice  = "ionice -c2 -n7 "
        # loop every 30 seconds at most
        watch = "watch -n60 "
        # silence the output and run in background
        # job   = ' 2>1 1>/dev/null'
        # build command
        command = watch + nice + rsync# + job
        FNULL = open(os.devnull, 'w')
        subprocess.Popen(command, shell=True, stdin=FNULL, stdout=FNULL, stderr=FNULL, close_fds=True)
        print(command)
        #os.system(command)
        os.system('reset')
    print('Launched background rsync processes')
    print('You may want this command later:')
    print("\tsudo ps aux | grep rsync | awk '{print $2}' | xargs kill -9")












if __name__=="__main__":

    # test encoding of function arguments as a string
    def example_function(a,b,c=1,d=('ok',),*vargs,**kw):
        ''' This docstring should be preserved by the decorator '''
        e,f = vargs if (len(vargs)==2) else (None,None)
        g = kw['k'] if 'k' in kw else None
        print(a,b,c,d,e,f,g)
    h = disk_cache(example_function)
    print('Testing argument siganture encoding')
    f   = example_function
    sig = ((('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')), None)
    for mode in ['repr','json','pickle','human']:
        print('\n',mode)
        print('\t',function_signature(f))
        print('\t',sig)
        try:
            fn  = signature_to_file_string(f,sig,mode=mode)
            print('\t',fn)
            s2  = file_string_to_signature(fn,mode=mode)
            print('\t',s2)
        except ValueError:
            traceback.print_exc()

    # test hierarchical cache
    print('Testing hybrid caching')
    print('Caution this is highly experimental')
    fn = leviathan(example_function)
    print('Testing leviathan ',fn.__name__)
    val = 'gamma'
    fn('a',val,'c','d')
    fn('a',val,'c','d','e','f')
    fn('a',val,c='c',d='d')
    fn('a',val,**{'c':'c','d':'d'})
    fn('a',val,*['c','d'])
    fn('a',val,d='d',*['c'])
    fn('a',val,*['c'],**{'d':'d'})
    fn('a',val,'c','d','e','f')
