#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Functions related to disk caching (memoization)
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import os,sys
__PYTHON_2__ = sys.version_info<(3, 0)

import numpy as np
import scipy.io
import inspect
import ast
import types
import time
import subprocess
import warnings
import traceback
import errno
import pickle
import json
import base64
import zlib
import hashlib
import shutil

from collections import defaultdict
from pickle import UnpicklingError

# TODO: we should use the same pickle library as multiprocessing uses
# for better comptability with parallelism and multiprocessing
try:
    from cPickle import PicklingError
except:
    from pickle import PicklingError

import neurotools.util.tools
import neurotools.jobs
import neurotools.jobs.ndecorator
from   neurotools.jobs.closure   import verify_function_closure
from   neurotools.jobs.filenames import is_dangerous_filename
from   neurotools.jobs.filenames import check_filename

from pathlib import Path

def get_source(f):
    '''
    Extracts and returns the source code of a function 
    (if it exists). 
    
    Parameters
    ----------
    f: function
        Function for which to extract source code
    
    Returns
    -------
    :str
        String containing the source code of 
        the passed function        
    '''
    g = neurotools.jobs.ndecorator.unwrap(f)
    try:
        return inspect.getsource(g)
    except (OSError,IOError):
        if hasattr(f,'__source__'): return f.__source__
        return inspect.getsource(f)
    raise ValueError('Cannot get function source')

@neurotools.jobs.ndecorator.memoize
def function_hash_no_subroutines(f):
    '''
    See ``function_hash_with_subroutines``. 
    This hash value is based on the

     1. Undecorated source code
     2. Docstring
     3. Function name
     4. Nodule name
     5. Function argument specification

    This function cannot detect changes in function behavior as a result of 
    changes in subroutines, global variables, or closures over mutable objects.
    
    Parameters
    ----------
    f: function
        Function for which to generate a hash value
    
    Returns
    -------
    :str
        Hash value that depends on the function. Hash is 
        constructed such that changes in function source 
        code and some dependencies will also generate a 
        different hash. 
    '''
    source    = get_source(f)
    docstring = inspect.getdoc(f)
    name      = f.__name__
    module    = f.__module__
    argspec   = neurotools.jobs.ndecorator.sanitize(
        inspect.getargspec(f))
    return hash((
        module,name,docstring,source,argspec,subroutines))

def function_signature(f):
    '''
    Generates string identifying the cache folder for function ``f``.

    We want to cache results to disk. However, these cached
    results will be invalid if the source code changes. It is hard
    to detect this accurately in Python. 
    Cache entries can also become invalid if the behavior of 
    subroutines change. To address this, the cache folder name includes
    a hash that depends on the function's
    
     - module,
     - name, 
     - argspec,
     - source, and
     - file.
     
    If any of these change, the chache folder will as well. This reduces
    the chances of retrieving stale / invalid cached results.

    Parameters
    ----------
    f: function

    Returns
    -------
    :str
        name+'.'+code
    
    '''
    # The one thing the decorator module can't fake is 
    # where the function is defined. So we can't see the 
    # source code directly if we're passed a wrapped 
    # function. We can however detect this case and peel
    # away the layers to get to the underlying source. The
    # decorator module will leave the wrapped function in a
    # variable  called __wrapped__, so we can follow this
    # back to the source code
    g = f
    source    = get_source(f)
    docstring = inspect.getdoc(f)
    name      = f.__name__
    module    = f.__module__
    
    try:
        argspec = inspect.getargspec(f)
    except DeprecationWarning:
        result    = inspect.getfullargspec(f)
        named     = result.args
        vargname  = result.varargs
        kwargname = result.varkw
        defaults  = result.defaults
        argspec = (named,vargname,kwargname,defaults)
        
    argspec   = neurotools.jobs.ndecorator.sanitize(argspec)
    identity  = (module,name)
    signature = (docstring,source,argspec)
    name = '.'.join(identity)
    code = base64hash10bytes((identity,signature))
    return name+'.'+code

def signature_to_file_string(f,sig,
    mode='repr',
    compressed=True,
    base64encode=True,
    truncate=True):
    '''
    Converts an argument signature to a string if possible. 
    
    This can be used to store cached results in a human-
    readable format. Alternatively, we may want to encode 
    the value of the argument signature in a string that is 
    compatible with most file systems.
    
    This does not append the file extension.
    
    Reasonable restrictions for compatibility:
    
     - No more than 4096 characters in path string
     - No more than 255 characters in file string
     - For windows compatibility try to limit it to 
       260 character total pathlength
     - These characters should be avoided: ``\/<>:"|?*,@#={}'&`!%$. ASCII 0..31``

    The easiest way to avoid problematic characters without
    restricting the input is to re-encode as base 64.

    **The following modes are supported:**

    **repr:** Uses ``repr`` and 
    ``ast.literal_eval(node_or_string)`` to serialize the 
    argument signature. This is safe, but restricts the 
    types permitted as paramteters.

    **json:** Uses json to serialize the argument signature. 
    Argument signatures cannot be uniquely recovered, 
    because tuples and lists both map to lists in the json 
    representation. Restricting the types used in the 
    argument signature may circumvent this.

    **pickle:** Uses pickle to serialize argument 
    signature. This should uniquely store argument 
    signatures that can be recovered, but takes more space. 
    Use this with caution, since changes to the pickle 
    serialization protocol between version will make the 
    encoded data irretrievable.

    **human:** Attempts a human-readable format.
    Experimental.

    Compression is on by defaut
    Signatures are base64 encoded by default
    
    Parameters
    ----------
    f: str
        Function being called
    sig:
        Cleaned-up function arguments created by
        ``neurotools.jobs.ndecorator.argument_signature()``
        A tuple of:
            args: tuple
                A tuple consisting of a list of
                (argument_name, argument_value) tuples.
            vargs:
                A tuple containing extra variable 
                arguments ("varargs"), if any.
    
    Other Parameters
    ----------------
    mode: str; default 'repr'
        Can be ``'repr'`` ``'json'`` ``'pickle'`` ``'human'``.
    compressed: boolean; default True
        Compress the resulting signature usingzlib?
    base64encode: boolean; default True
        Base-64 encode the resulting signature?
    truncate: boolean; default True
        Truncate file names that are too long?
        This will discard data, but the truncated signature
        may still serve as an identified with a low 
        collision probability.
        
    Returns
    -------
    filename: str
    '''
    sig = neurotools.jobs.ndecorator.sanitize(sig)

    if compressed and not base64encode:
        raise ValueError(
        'To use compression set base64encode=True')

    # A hash value gives us good distribution to control
    # the complexity of the directory tree used to manage
    # the cache, but is not unique
    hsh = base64hash10bytes(sig)    

    # We also need to store some information about which
    # function this is for. We'll get a human readable
    # name identifying the funciton, and a shorter
    # hash-value to make sure we invalidate the cache if
    # the source code or function definition changes.
    fname = function_signature(f)

    # The argument spec can be mapped uniquely to a file 
    # name by converting it to text, then converting this
    # text to base64 to avoid issues with special
    # characters. Passing the text representation through
    # zlib preserves the uniqueness of the key, while
    # reducing the overall size. This improves performance
    # convert key to an encoded string
    if   mode=='repr'  : key = repr(sig)
    elif mode=='json'  : key = json.dumps(sig)
    elif mode=='pickle': key = pickle.dumps(sig)
    elif mode=='human' : key = human_encode(sig)
    else: raise ValueError(
        'I support coding modes repr, json, and pickle. '
        'I don\'t recognize coding mode %s'%mode)
    # compress and base64 encode string
    key = key.encode('UTF-8')
    if compressed  : key = zlib.compress(key)
    if base64encode: key = base64.urlsafe_b64encode(key)

    # Path will be a joining of the hash and the key. The 
    # hash should give good distribution, while the key 
    # means we can recover the arguments from the file name.
    filename = '%s.%s.%s'%(fname,hsh,key.decode())
    # If for some reason the path is too long, complain
    if len(filename)>255:
        if truncate:
            # hash the key if it is too long and truncation 
            # is enabled
            s  = key.decode()
            kh = base64hash(s)            
            filename = '%s.%s.%s'%(fname,hsh,kh)
            filename = filename[:255]
        else: raise ValueError(
            'Argument specification exceeds maximum path '
            'length. Function probably accepts data as an '
            'argument, rather than a key to locate data. '
            'See Joblib for a caching framework that uses '
            'cryptographic hashes to solve this problem. '
            'For now, we skip the cache. The offending '
            'filename is '+filename)
    if __PYTHON_2__:
        try:
            ascii = filename.encode("utf8","ignore")
            assert unicode(ascii)==filename
            filename = ascii
        except UnicodeDecodeError:
            pass
    check_filename(filename)
    return filename

def file_string_to_signature(
    filename,
    mode='repr',
    compressed=True,
    base64encode=True):
    '''
    Extracts the argument key from the compressed 
    representation in a cache filename entry. Inverse of 
    ``signature_to_file_string()``.
    
    The ``filename`` should be provided as a string, without
    the file extension.

    The following modes are supported:

    **repr:** Uses repr and 
    ast.literal_eval(node_or_string) to serialize the 
    argument signature. This is safe, but restricts the 
    types permitted as paramteters.

    **json:** Uses json to serialize the argument signature. 
    Argument signatures cannot be uniquely recovered, 
    because tuples and lists both map to lists in the json 
    representation. Restricting the types used in the 
    argument signature may circumvent this.

    **pickle:** Uses pickle to serialize argument 
    signature. This should uniquely store argument 
    signatures that can be recovered, but takes more space. 
    Use this with caution, since changes to the pickle 
    serialization protocol between version will make the 
    encoded data irretrievable.

    **human:** Attempts a human-readable format. 
    Experimental.

    human:
        Attempts a human-readable format. Experimental.

    Compression is on by default
    Signatures are base64 encoded by default
    
    Parameters
    ----------
    filename: str
        Encoded filename, as a string, *without* the file
        extension 
    
    Other Parameters
    ----------------
    mode: str; default 'repr'
        Can be ``'repr'`` ``'json'`` ``'pickle'`` ``'human'``.
    compressed: boolean; default True
        Whether ``zlib`` was used to compress this function
        call signature
    base64encode: boolean; default  True
        Whether this function call signature was base-65
        encoded.
    
    Returns
    -------
    sig: nested tuple
        Function arguments created by
        ``neurotools.jobs.ndecorator.argument_signature()``
        A tuple of:
            args: tuple
                A tuple consisting of a list of
                (argument_name, argument_value) tuples.
            vargs:
                A tuple containing extra variable 
                arguments ("varargs"), if any.
    '''
    pieces = filename.split('.')
    key    = pieces[-1]
    hsh    = pieces[-2]
    name   = '.'.join(pieces[:-3])

    #try:
    # The argument spec can be mapped uniquely to a file
    # name by converting it to text, then converting 
    # this text to base64 to avoid issues with special 
    # characters. Passing the text representation 
    # through zlib preserves the uniqueness of the key, 
    # while reducing the overall size. This improves
    # performance.
    if base64encode: key = base64.urlsafe_b64decode(
        (key+'='*10).encode('UTF-8'))
    if compressed  : key = zlib.decompress(key)
    key = key.decode()
    if   mode=='repr'  : sig = ast.literal_eval(key)
    elif mode=='json'  : sig = json.loads(key)
    elif mode=='pickle': sig = pickle.loads(key)
    elif mode=='human' : sig = human_decode(key)
    else: raise ValueError((
        'I support coding modes repr, json, and pickle;'
        ' I don\'t recognize coding mode %s')%mode)
    sig = neurotools.jobs.ndecorator.sanitize(sig)
    return sig
    '''
    except:
        raise ValueError((
            'Could not decode "%s"; Please ensure that you'
            'provide the file name without the file '
            'extension')%filename)
    '''
    
    
def human_encode(sig):
    '''
    Formats an argument signature for saving as file name
    
    Parameters
    ----------
    sig: nested tuple
        Argument signature as a safe nested tuple
    
    Returns
    -------
    result: str
        Human-readable argument-signature filename
    '''
    sig = neurotools.jobs.ndecorator.sanitize(
        sig,mode='strict')
    named, vargs = sig
    if not vargs is None:
        raise ValueError(
            'Currently variable arguments are not permitted'
            ' in the human-readable format')
    result = ','.join(
        ['%s=%s'%(k,repr(v)) for (k,v) in named])
    return result

def human_decode(key):
    '''
    Formats the argument signature for saving as file name
    
    Parameters
    ----------
    key: str
        Human-readable argument-signature filename
    
    Returns
    -------
    sig: nested tuple
        Argument signature as a nested tuple
    '''
    params = [k.split('=') for k in key.split(',')]
    params = tuple((n,ast.literal_eval(v)) for n,v in params)
    sig = (params,None)
    sig = neurotools.jobs.ndecorator.sanitize(
        sig,mode='strict')
    return sig

def get_cache_path(cache_root,f,*args,**kwargs):
    '''
    Locate the directory path for function ``f`` within the
    ``__neurotools_cache__`` path ``cache_root``.
    
    Parameters
    ----------
    cache_root: str
        Path to root of the ``__neurotools__`` cache
    f: function
        Cached function object
        
    Returns
    -------
    path: str
    '''
    sig = neurotools.jobs.ndecorator.argument_signature(
        f,*args,**kwargs)
    fn  = signature_to_file_string(f,sig,
        mode        ='repr',
        compressed  =True,
        base64encode=True)
    pieces = fn.split('.')
    # first two words used as directories
    path = cache_root + os.sep + os.sep.join(pieces[:-2]) + os.sep
    return path

def locate_cached(cache_root,f,method,*args,**kwargs):
    '''
    Locate a specific cache entry within ``cache_root`` for
    function ``f`` cached with method ``method``, and called
    with arguments ``*args`` and keyword arguments ``**kwargs``.
    
    Parameters
    ----------
    cache_root: str
        directory/path as string
    f: function
        Function being cached
    method: str
        Cache file extension e.g. ``"npy"``, "``mat``", etc. 
    args: iterable
        function parameters
    kwargs: dict
        function keyword arguments
    
    Returns
    -------
    fn: str   
        File name of cache entry without extension
    sig: tuple
        Tuple of (args,kwargs) info from 
        ``argument_signature()``
    path: str
        Directory containing cache file    
    filename: str
        File name with extension
    location: str
        Full absolute path to cache entry
    '''
    while method.startswith('.'): method=method[1:]
    sig = neurotools.jobs.ndecorator.argument_signature(f,*args,**kwargs)
    fn  = signature_to_file_string(f,sig,
        mode        ='repr',
        compressed  =True,
        base64encode=True)

    pieces = fn.split('.')
    # first two words used as directories
    path = cache_root + os.sep + os.sep.join(pieces[:-2]) + os.sep
    # remaining pieces a filename
    filename = '.'.join(pieces[-2:])+'.'+method
    location = path+filename
    return fn,sig,path,filename,location

def validate_for_matfile(x):
    '''
    Verify that the nested tuple ``x``, which contains the
    arguments to a function call, can be safely stored 
    in a Matlab matfile (``.mat``).
    
    .. table:: Numpy types: these should be compatible
        :widths: auto
        
        ==========  ========================================
        Type        Description
        ==========  ========================================
        bool 	    Boolean (True or False) stored as a byte
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
        complex64 	Complex number, represented by two float32
        complex128 	Complex number, represented by two float64
        ==========  ========================================
    
    
    Parameters
    ----------
    x: nested tuple
        Arguments to a function
    
    Returns
    -------
    :boolean
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

import warnings

def validate_for_numpy(x):
    '''
    Check whether an array-like object can safely be stored 
    in a numpy archive. 
    
    .. table:: Numpy types: these should be compatible
        :widths: auto
        
        ==========  ========================================
        Type        Description
        ==========  ========================================
        bool 	    Boolean (True or False) stored as a byte
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
        complex64 	Complex number, represented by two float32
        complex128 	Complex number, represented by two float64
        ==========  ========================================
    
    
    Parameters
    ----------
    x: object
        array-like object; 
    
    Returns
    -------
    :boolean
        True if the data in ``x`` can be safely stored in a 
        Numpy archive
    '''
    safe = (
        np.bool_  , np.int8     , np.int16 , np.int32 , np.int64  ,
        np.uint8  , np.uint16   , np.uint32, np.uint64, np.float32,
        np.float64, np.complex64, np.complex128)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
    if not isinstance(x,np.ndarray):
        try:
            x = np.array(x)
        except:
            x = [*x]
            _x = np.empty(len(x),dtype=object)
            for i,xi in enumerate(x):
                _x[i] = x[i]
            x = _x
    if x.dtype == object:
        # object arrays will be converted to cell arrays,
        # we need to make sure each cell can be stored safely
        try:
            ix = iter(x)
        except TypeError as te:
            raise ValueError('is not iterable')
        return map(validate_for_numpy,x)
    if not x.dtype in safe:
        raise ValueError("Numpy type %s is not on the list"
            " of compatible types"%x.dtype)
    return True

def read_cache_entry(location,method):
    if method=='pickle':
        with open(location,'rb') as openfile:
            return pickle.load(openfile)
    elif method =='mat':
        return scipy.io.loadmat(location)['varargout']
    elif method =='npy':
        return np.load(location,allow_pickle=True)

def disk_cacher(
    cache_location,
    method     = 'npy',
    write_back = True,
    skip_fast  = False,
    verbose    = False,
    allow_mutable_bindings=False,
    cache_identifier='__neurotools_cache__'):
    '''
    Decorator to memoize functions to disk.
    Currying pattern here where cache_location creates 
    decotrators.

    write_back:

         True: Default. Computed results are saved to disk

        False: Computed results are not saved to disk. In 
               this case of hierarchical caches mapped to 
               the filesystem, a background rsync loop can 
               handle asynchronous write-back.

    method:

         p: Use pickle to store cache. Can serialize all 
            objects but seriously slow! May not get ANY 
            speedup due to time costs if pickling and disk 
            IO.

       mat: Use scipy.io.savemat and scipy.io.loadmat. 
            Nice because it's compatible with matlab. 
            Unfortunately, can only store numpy types and 
            data that can be converted to numpy types. Data 
            conversion may alter the types of the return 
            arguments when retrieved from the cache.

       npy: Use built in numpy.save functionality. 

      hdf5: Not yet implemented.
      
    
    Parameters
    ----------
    cache_location: str
        Path to disk cache
        
    Other Parameters
    ----------------
    method: str; default 'npy'
        Storange format for caches. 
        Can be 'pickle', 'mat' or 'npy'
    write_back: boolean; default True
        Whether to copy new cache value back to the disk 
        cache. If false, then previously cached values can 
        be read but new entries will not be creates
    skip_fast: boolean; default False
        Attempt to simply re-compute values which are 
        taking too long to retrieve from the cache. 
        Experimental, do not use. 
    verbose: boolean; default False
        Whether to print detailde logging information
    allow_mutable_bindings: boolean; default False
        Whether to allow caching of functions that close 
        over mutable scope. Such functions are more likely
        to return different results for the same arguments, 
        leading to invalid cached values.
    cache_identifier: str; default 'neurotools_cache'
        subdirectory name for disk cache.
    
    Returns
    -------
    cached : disk cacher object
        TODO
    '''
    VALID_METHODS = ('pickle','mat','npy')
    assert method in VALID_METHODS
    cache_location = os.path.abspath(cache_location)+os.sep
    cache_root     = cache_location+cache_identifier
    neurotools.util.tools.ensure_dir(cache_location)
    neurotools.util.tools.ensure_dir(cache_root)
    
    def cached(f):
        '''
        The ``disk_cacher`` function constructs a decorator 
        ``cached`` that can be used to wrap functions to 
        memoize their results to disk. ``cached`` returns the
        ``decorated`` object which is constructed by
        calling the inner function ``wrapped``.
        
            cached <-- disk_cacher(location,...)
            caching_function <-- cached(somefunction)
        
        '''
        if not allow_mutable_bindings:
            verify_function_closure(f)
            
        # Patch for 2/3 compatibility
        if __PYTHON_2__:
            FileError = IOError
        else:
            FileError = FileNotFoundError
            
        @neurotools.jobs.ndecorator.robust_decorator
        def wrapped(f,*args,**kwargs):
            '''
            This is a wrapper for memoizing results to disk. 
            This docstring should be overwritten by the 
            docstring of the wrapped function.
            '''
            t0 = neurotools.util.time.current_milli_time()

            # Store parameters;
            # These will be saved in the cached result
            params = (args,tuple(list(kwargs.items())))
            try:
                fn,sig,path,filename,location = \
                    locate_cached(
                        cache_root,f,method,*args,**kwargs)
            except ValueError as exc:
                print('Generating cache key failed')
                traceback.print_exc()#exc)
                time,result = f(*args,**kwargs)
                return result
            
            result = None
            if os.path.isfile(location):
                try:
                    result = read_cache_entry(location,method)
                    if verbose:
                        print('Retrieved cache at ',path)
                        print('  %s.%s'%(f.__module__,f.__name__))
                        print('  %s'%neurotools.jobs.ndecorator.print_signature(sig))
                except (ValueError, EOFError, OSError, IOError, FileError, UnpicklingError) as exc:
                    if verbose: print('  File reading failed')

            if not result is None:
                params,result = result
            else:
                if verbose:
                    print('Recomputing cache at %s'%cache_location)
                    print('  %s.%s'%(f.__module__,f.__name__))
                    print('  %s'%neurotools.jobs.ndecorator.print_signature(sig))

                # Evaluate function
                time,result = f(*args,**kwargs)
                if verbose:
                    print('  %s'%path)
                    print('  Took %d milliseconds'%time)

                # Save Cached output to disk
                if write_back:
                    savedata = (params,result)
                    neurotools.util.tools.ensure_dir(path)
                    Path(location).touch()
                    if verbose: print('Writing cache at ',path)
                    try:
                        if method=='pickle':
                            with open(location,'wb') as openfile:
                                pickle.dump(savedata,openfile,protocol=pickle.HIGHEST_PROTOCOL)
                        elif method =='mat':
                            validated_result = validate_for_matfile(savedata)
                            if validated_result is None:
                                raise ValueError(
                                    'Error: return value cannot be safely packaged in a matfile')
                            scipy.io.savemat(location,{'varargout':savedata})
                        elif method =='npy':
                            validated_result = validate_for_numpy(savedata)
                            if validated_result is None:
                                raise ValueError(
                                    'Error: return value cannot be safely packaged in a numpy file')
                            sd = np.empty(2,dtype=object)
                            sd[0] = savedata[0]
                            sd[1] = savedata[1]
                            np.save(location, sd)
                    except (ValueError, IOError, PicklingError) as exc2:
                        if verbose:
                            print('Saving cache at %s FAILED'%cache_location)
                            print('  %s.%s'%(f.__module__,f.__name__))
                            print('  %s'%\
                                neurotools.jobs.ndecorator.print_signature(sig))
                            print('  '+'\n  '.join(\
                                traceback.format_exc().split('\n')))

                    if verbose:
                        try:
                            print('Wrote cache at ',path)
                            print('  For function %s.%s'%\
                                (f.__module__,f.__name__))
                            print('  Argument signature %s'%\
                                neurotools.jobs.ndecorator.print_signature(sig))
                            st        = os.stat(location)
                            du        = st.st_blocks * st.st_blksize
                            t1        = neurotools.util.time.current_milli_time()
                            overhead  = float(t1-t0) - time
                            io        = float(du)/(1+overhead)
                            recompute = float(du)/(1+time)
                            boost     = (recompute-io)
                            saved     = time - overhead
                            quality   = boost/(1+float(du))
                            print('  Size on disk is %d'%du)
                            print('  IO overhead %d milliseconds'%overhead)
                            print('  Cached performance %0.4f'%io)
                            print('  Recompute cost     %0.4f'%recompute)
                            print('  Expected boost     %0.4f'%boost)
                            print('  Time-space quality %0.4f'%quality)
                        except (OSError) as exc3:
                            print('\n  '.join(\
                                traceback.format_exc().split('\n')))
                    # Skipping when the cache is slower than recompute is not yet supported
                    # if skip_fast and boost<0:
                    #    if verbose:
                    #        print('  WARNING DISK IO MORE EXPENSIVE THAN RECOMPUTING!')
                    #        print('  We should really do something about this?')
                    #        print('  Zeroing out the file, hopefully that causes it to crash on load?')
                    #    with open(location, 'w'): pass
            return result
        
        def purge(*args,**kwargs):
            '''
            Delete cache entries matching arguments. This is
            a destructive operation, execute with care.
    
            Parameters
            ----------
            *args
                Arguments forward to the ``locate_cached`` 
                function. Matching cache entries will be 
                deleted.
            **kwargs
                Keyword arguments forward to the 
                ``locate_cached`` function Matching cache 
                entries will be deleted.
            '''
            for method in VALID_METHODS:
                fn,sig,path,filename,location = \
                    locate_cached(
                        cache_root,f,method,*args,**kwargs)
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
        
        def lscache(verbose=False):
            '''
            List all files associated with cached 
            invocations of the wrapped function.
            ("cache entries")
            '''
            path = cache_root + os.sep +\
                os.sep.join(
                    function_signature(f).split('.'))
            try:
                files = os.listdir(path)
            except:
                files = []
            if verbose:
                print('Cache %s contains:'%path)
                print('\n  '+'\n  '.join([
                    f[:20]+'…' for f in files
                ]))
            return path,files
        
        @neurotools.jobs.ndecorator.robust_decorator
        def locate(f,*args,**kwargs):
            '''
            A version of the decorator that simply locates 
            the cache file. The result of ``locate_cached`` is
            returned directly. It is a tuple:
            
                (fn,sig,path,filename,location)
            
            Returns
            -------
            fn: str   
                File name of cache entry without extension
            sig: tuple
                Tuple of (args,kwargs) info from 
                ``argument_signature()``
            path:str
                Directory containing cache file    
            filename: str  
                File name with extension
            location: str  
                Full absolute path to cache entry
            '''
            return locate_cached(cache_root,f,method,*args,**kwargs)
        
        # Bulid decorated function and
        # Save additional methods associated with decorated object
        decorated            = wrapped(neurotools.jobs.ndecorator.timed(f))
        decorated.purge      = purge
        decorated.cache_root = cache_root
        decorated.lscache    = lscache
        decorated.locate     = locate(f)
        return decorated
    
    cached.cache_root = cache_root
    return cached


def hierarchical_cacher(fast_to_slow,
        method='npy',
        write_back=True,
        verbose=False,
        allow_mutable_bindings=False,
        cache_identifier ='neurotools_cache'):
    '''
    Construct a filesystem cache defined in terms of a 
    hierarchy from faster to slower (fallback) caches.
    
    Parameters
    ----------
    fast_to_slow : tuple of strings
        list of filesystem paths for disk caches in order 
        from the fast (default or main) cache to slower.
        
    Other Parameters
    ----------------
    method: string, default ``'npy'``
        cache storing method;
    write_back : bool, default True
        whether to automatically copy newly computed cache 
        values to the slower caches
    verbose : bool, defaults to ``False``
        whether to print detailed logging iformation to 
        standard out when manipulating the cache
    allow_mutable_bindings : bool, default False
        If true, then "unsafe" namespace bindings, for 
        example user-defined functions, will be allowed in 
        disk cached functions. If a cached function calls 
        subroutines, and those subroutines change, the disk
        cacher cannot detect the implementation different.
        Consequentially, it cannot tell whether old cached 
        values are invalid. 
    cache_identifier : str, default 'neurotools_cache'
        (sub)folder name to store cached results
    
    Returns
    -------
    hierarchical: decorator
        A hierarchical disk-caching decorator that can be 
        used to memoize functions to the specified disk 
        caching hierarchy. 
    '''
    slow_to_fast = fast_to_slow[::-1] # reverse it
    all_cachers  = []
    def hierarchical(f):
        # disable write-back on the slow caches
        for location in slow_to_fast[:-1]:
            f = disk_cacher(location,
                method                 = method,
                write_back             = write_back,
                verbose                = verbose,
                allow_mutable_bindings = allow_mutable_bindings,
                cache_identifier       = cache_identifier)(f)
            all_cachers.append(f)
        # use write-back only on the fast cache
        location = slow_to_fast[-1]
        f = neurotools.jobs.cache.disk_cacher(location,
            method                 = method,
            write_back             = True,
            verbose                = verbose,
            allow_mutable_bindings = allow_mutable_bindings,
            cache_identifier       = cache_identifier)(f)
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


def scan_cachedir(
    cachedir,
    method="npy",
    verbose=False,
    **kw):
    '''
    Retrieve all entries in ``cachedir``, unpacking their 
    encoded arguments.
    
    Parameters
    ----------
    cachedir: str
        Cache directory to scan, e.g. 
        ``__neurotools_cache__/…/…/…/somefunction``
    
    Other Parameters
    ----------------
    method: str; default ``'npy'``
        Can be ``'npy'`` or ``'mat'``
    verbose: boolean; default False
    **kw:
        Forwarded to ``file_string_to_signature()``; 
        See ``file_string_to_signature()`` for details.
        
    Returns
    -------
    :dict
        ``filename -> (args,varags)`` dictionary, where
        ``args`` is a ``parameter_name -> value`` dictionary
        and ``varargs`` is a list of extra arguments, if 
        any.
    '''
    if not method.startswith('.'):
        method = '.'+method

    argnames = None
    results  = {}
    invalid  = []
    for f in os.listdir(cachedir):
            name, ext = os.path.splitext(f)
            if not ext==method: continue
            
            # If this fails we can try to recover from the
            # cached contents
            try: 
                args, varargs = file_string_to_signature(
                    name,**kw)
                
                if len(args)==2 and isinstance(args[0],str):
                    args = (args,)

                # Remember argument names, we might need 
                # these to recover signatures from files  
                # whose filename-based decoding fails
                _argnames,_ = zip(*args)
                if argnames is None:
                    argnames = _argnames
                elif not argnames==_argnames:
                    raise ValueError(('File %s argument '
                        'names %s differs from previous '
                        'argument names %s')%(
                        f,_argnames,argnames))

                # Save arguments as dictionary
                args = dict(args)
                results[f] = (args,varargs)
            except zlib.error as e:
                invalid.append(f)
    
    if len(invalid):
        if verbose:
            warnings.warn(
                'The following files could not be decoded:'+
                '\n    '+'\n    '.join(invalid))
        else:
            warnings.warn(
                '%d files could not be decoded'%\
                len(invalid))
            
        # Try to recover
        if method=='.npy':
            if argnames is None:
                raise ValueError('No valid reference cache '
                    'entry was available for identifying '
                    'the function arguments; I would need '
                    'the original function used to produce '
                    'this cache to proceed.')
            warnings.warn(
                'Format is .npy; I will try recover'
                ' by inspecting file contents')
            double_failed = []
            for f in invalid:
                try:
                    args, varargs = np.load(
                        cachedir+os.sep+f,allow_pickle=True
                    )[0]
                    args = dict(zip(argnames,args))
                    results[f] = (args,varargs)
                except:
                    double_failed.append(f)
            warnings.warn(
                '%d/%d recovered'%(
                    len(invalid)-len(double_failed),
                    len(invalid))
            )
            if len(double_failed):
                warnings.warn(
                    '%d files irrecoverable'%\
                    len(double_failed))

    return results
    
def hashit(obj):
    if not isinstance(obj,bytes):
        try:
            obj = obj.encode('UTF-8')
        except:
            obj = repr(obj).encode('UTF-8')
    return hashlib.sha224(obj).digest()#[::-1]

def base64hash(obj):
    '''
    Retrieve a base-64 encoded hash for an object.
    This uses the built-in ``encode`` function to convert an object to
    ``utf-8``, then calls ``.sha224(obj).digest()`` to create a hash,
    finally packaging the result in base-64.
    
    Parameters
    ----------
    obj: object
    
    Returns
    -------
    code: str
    '''
    code = base64.urlsafe_b64encode(hashit(obj)).decode().replace('=','')
    #code = base64.urlsafe_b64encode(str(hashit(obj)).encode('UTF-8')).decode().replace('=','')
    return code

def base64hash10bytes(obj):
    '''
    Retrieve first two bytes of a base-64 encoded has for 
    an object.
    
    Parameters
    ----------
    obj: object
    
    Returns
    -------
    code: str
    '''
    code = base64.urlsafe_b64encode(hashit(obj)[:10]).decode().replace('=','')
    #code = base64.urlsafe_b64encode(str(hashit(obj)).encode('UTF-8')).decode().replace('=','')
    return code
    
@neurotools.jobs.ndecorator.memoize
def function_hash_with_subroutines(f,force=False):
    '''
    Functions may change if their subroutines change. This 
    function computes a hash value that is sensitive to 
    changes in the source code, docstring, argument 
    specification, name, module, and subroutines.

    This is a recursive procedure with a fair amount of 
    overhead. To allow for the possibility of mutual 
    recursion, subroutines are excluded from the hash if 
    the function has already been visited.

    This does not use the built-in hash function for 
    functions in python.

    **Ongoing development notes**

    *Is memoization possible?* Making memoization compatible 
    with graceful handling of potentially complex mutually 
    recurrent call structures is tricky. Each function 
    generates a call tree, which does not expand a node if
    it is already present in the call tree structure. 
    Therefore there are many possible hash values for an 
    intermediate function depending on how far it's call 
    tree gets expanded, which depends on what has been 
    expanded and encountered so far. Therefore, we cannot
    cache these intermediate values.

    *Note:* the topology of a mutually recurrent call 
    structure cannot change without changing the source 
    code of at least one function in the call graph? 
    So it suffices to (1) hash the subroutines, (2) 
    expand the call graph (potentially excluding standard 
    and system library functions), (3) grab the non-
    recursive hash for each of these functions, 
    and (4) then generate the subroutine dependent hash by 
    combining the non-recursive hash with the hash of a 
    datastructure representing the subroutine "profile" 
    obtained from the call graph.
    
    We assume that any decorators wrapping the function do 
    not modify it's computation, and can safely be stripped.

    Note that this function cannot detect changes in 
    effective function behavior that result from changes 
    in global variables or mutable scope that has been 
    closed over.
    
    Parameters
    ----------
    force: boolean
        force muse be true, otherwise this function will 
        fail with a warning. 
    
    Returns
    -------
    :str
        Hash of function
    '''
    if not force:
        raise NotImplementedError(
        'It is not possible to hash a function reliably')

    # repeatedly expand list of subroutines
    to_expand = {f}
    expanded  = set()
    while len(to_expand)>0:
        new_subroutines = set()
        for g in to_expand: 
            new_subroutines|=get_subroutines(g)
        expanded |= to_expand
        to_expand = new_subroutines - expanded
    # we now have a set, we need to provide some ordering 
    # over that set sort the hash values and hash that
    return hash(tuple(sorted(map(
        function_hash_no_subroutines,expanded))))
        
def combine_caches(cache_root,f):
    '''
    Merge all cache folders for function ``f`` 
    by copying cache files into the current cache folder.
    
    Usually, the existence of multiple cache folders 
    indicates that cache files were generated using 
    versions of ``f`` with different source code. However, 
    you may want to merge caches if you are certain that 
    such changes code did not change the function's 
    behavior.
    
    Parameters
    ----------
    cache_root: str
        path to the top-level cache directory
    f: function
        cached function to merge
    '''
    fs = function_signature(f)
    copy_to = fs.split('.')[-1]
    parent = os.path.join(
        cache_root,
        os.sep.join(fs.split('.')[:2]))
    copy_from = {*os.listdir(parent)} - {copy_to}
    for fr in copy_from:
        for fn in os.listdir(parent+os.sep+fr):
            fto = parent+os.sep+copy_to+os.sep+fn
            ffr = parent+os.sep+fr+os.sep+fn
            if not os.path.exists(fto):
                shutil.copy2(ffr,fto)
    return copy_to
                
def exists(cache_root,f,method,*args,**kwargs):
    '''
    Check if a cached result for ``f(*args,**kwargs)`` 
    of type ``method`` exists in cache ``cache_root``.
    
    Parameters
    ----------
    cache_root: str
        directory/path as string
    f: function
        Function being cached
    method: str
        Cache file extension e.g. ``"npy"``, "``mat``", etc. 
    args: iterable
        function parameters
    kwargs: dict
        function keyword arguments
    
    Returns
    -------
    :boolean
        True if the cache file exists
    '''
    return os.path.exists(
        locate_cached(
            cache_root,f,method,*args)[-1])
                
                
                

