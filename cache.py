#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

import scipy.io
import neurotools.decorator
import neurotools.tools
import os,sys
from collections import defaultdict
import inspect, ast, types
import os, time, sys, subprocess
import warnings, traceback, errno
import pickle, json, base64, zlib
from neurotools.ntime import current_milli_time

CACHE_IDENTIFIER ='.__neurotools_cache__'

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
    if type(filename) is unicode: 
        return True
    if any([c in filename for c in "\/<>:\"'|?*,@#{}'&`!%$\n\t\b\r "]):
        return True
    return False

def check_filename(filename):
    if len(filename)>255: 
        warnings.warn('FILE NAME MAY BE TOO LONG ON SOME SYSTEMS')
    if type(filename) is unicode: 
        warnings.warn('FILE NAME IS UNICODE')
    if any([c in filename for c in "/?<>\:*|\"\n\t\b\r"]):
        raise ValueError('Filename contains character forbidden on windows')
    if any([c in filename for c in "\/<>:\"'|?*,@#{}'&`!%$\n\t\b\r "]):
        warnings.warn('FILE NAME CONTAINS CHARACTER THAT MAY CAUSE ISSUES IN SOME SOFTWARE')

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
    while hasattr(g, '__wrapped__'): 
        g = g.__wrapped__
    source    = inspect.getsource(g)
    docstring = inspect.getdoc(f)
    name      = f.func_name
    module    = f.__module__
    argspec   = neurotools.decorator.sanitize(inspect.getargspec(f))
    identity  = (module,name)
    signature = (docstring,source,argspec)
    name = '.'.join(identity)
    code = base64.urlsafe_b64encode(\
        str(hash((identity,signature))&0xffff)).replace('=','')
    return name+'.'+code

def signature_to_file_string(f,sig,mode='repr',compressed=True,base64encode=True):
    '''
    Converts an argument signature to a string if possible. This can be 
    used to store cached results in a human-readable format. Alternatively, 
    we may want to simply encode the value of the argument signature in
    a string that is compatible with most file systems. We'd still need
    to perform verification on the object to 
    
    No more than 4096 characters in path string
    No more than 255 characters in file string
    For windows compatibility try to limit it to 260 character total pathlen
    
    For compatibility, these characters should be avoided in paths:
        \/<>:"|?*,@#={}'&`!%$. ASCII 0..31
        
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
    sig = neurotools.decorator.sanitize(sig)
    
    if compressed and not base64encode:
        raise ValueError('If you want compression, turn on base64 encoding')
    
    # A hash value gives us good distribution to control the complexity of
    # the directory tree used to manage the cache, but is not unique
    hsh = base64.urlsafe_b64encode(str(hash(sig)&0xffff)).replace('=','')

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
    if   mode=='repr'  : key = repr(sig)
    elif mode=='json'  : key = json.dumps(sig)
    elif mode=='pickle': key = pickle.dumps(sig)
    elif mode=='human' : key = human_encode(sig)
    else: raise ValueError('I support coding modes repr, json, and pickle\n'+
        'I do not recognize coding mode %s'%mode)
    if compressed  : key = zlib.compress(key)
    if base64encode: key = base64.urlsafe_b64encode(key)
    
    # Path will be a joining of the hash and the key. The hash should give
    # good distribution, while the key means we can recover the arguments
    # from the file name.
    filename = '%s.%s.%s'%(fname,hsh,key)
    # If for some reason the path is too long, complain
    if len(filename)>255:
        raise ValueError(\
            'Argument specification exceeds maximum path length.')
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
    if base64encode: key = base64.urlsafe_b64decode(key)
    if compressed  : key = zlib.decompress(key)
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
        raise ValueError('Currently variable arguments are not permitted '+
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
    sig = neurotools.decorator.argument_signature(f,*args,**kwargs)
    fn  = signature_to_file_string(f,sig,
            mode        ='repr',
            compressed  =True,
            base64encode=True)
    pieces   = fn.split('.')
    path     = cache_root + os.sep + os.sep.join(pieces[:-2]) + os.sep
    filename = '.'.join(pieces[-2:])+'.'+method
    location = path+filename
    return fn,sig,path,filename,location

def disk_cacher(cache_location,method='mat',write_back=True):
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
    assert method in ('pickle','mat','numpy')
    cache_location = os.path.abspath(cache_location)+os.sep
    cache_root     = cache_location+CACHE_IDENTIFIER
    def cached(f):
        @neurotools.decorator.robust_decorator
        def wrapped(f,*args,**kwargs):
            t0 = current_milli_time()
            fn,sig,path,filename,location = locate_cached(cache_root,f,method,*args,**kwargs)
            try:
                if method=='pickle':
                    result = pickle.load(open(location,'rb'),protocol=pickle.HIGHEST_PROTOCOL)
                elif method =='mat':
                    result = scipy.io.loadmat(location)['varargout']
                elif method =='npy':
                    result = numpy.load(location, mmap_mode='r', allow_pickle=True)
                                        
                print('Retrieved cache at ',path)
                print('\tFor function %s.%s'%(f.__module__,f.func_name))
                print('\tArgument signature %s'%neurotools.decorator.print_signature(sig))  
            except:
                print('Recomputing cache at %s'%cache_location)
                print('\tFor function %s.%s'%(f.__module__,f.func_name))
                print('\tArgument signature %s'%neurotools.decorator.print_signature(sig)) 
                
                time,result = f(*args,**kwargs)
                neurotools.tools.ensure_dir(path)
                
                print('\tTook %d milliseconds'%time)
                
                if write_back:
                    
                    if method=='pickle':
                        pickle.dump(result,open(location,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
                    elif method =='mat':
                        scipy.io.savemat(location,{'varargout':result})
                    elif method =='npy':
                        numpy.save(location, result, allow_pickle=True)
                    
                    st        = os.stat(location)
                    du        = st.st_blocks * st.st_blksize
                    t1        = current_milli_time()
                    overhead  = float(t1-t0) - time
                    io        = float(du)/(1+overhead)
                    recompute = float(du)/(1+time)
                    boost     = (recompute-io)
                    saved     = time - overhead
                    quality   = boost/float(du)
                    
                    print('Wrote cache at ',path)    
                    print('\tFor function %s.%s'%(f.__module__,f.func_name))
                    print('\tArgument signature %s'%neurotools.decorator.print_signature(sig)) 
                    print('\tSize on disk is %d'%du)
                    print('\tIO overhead %d milliseconds'%overhead)
                    print('\tCached performance %0.4f'%io)
                    print('\tRecompute cost     %0.4f'%recompute)
                    print('\tExpected boost     %0.4f'%boost)
                    print('\tTime-space quality %0.4f'%quality)
                    
                    if boost<0:
                        print('\tWARNING DISK IO MORE EXPENSIVE THAN RECOMPUTING!')
                        print('\tWe should really do something about this?')
                        print('\tZeroing out the file, hopefully that causes it to crash on load?')
                        with open(location, 'w'): pass
                
            return result
        return wrapped(neurotools.decorator.timed(f))
    return cached

disk_cached = disk_cacher('.')

def hierarchical_cacher(*fast_to_slow):
    '''
    Designed for constructing a hierarchy of disk caches. For now, only
    the fastest cache has write-back enabled. 
    An asynchronous rsync process can be used to synchronize the fast
    cache with slower caches
    '''
    slow_to_fast = fast_to_slow[::-1] # reverse it
    def hierarchical(f):
        # disable write-back on the slow caches
        for location in slow_to_fast[:-1]: 
            f = disk_cacher(location,write_back=False)(f)
        # use write-back only on the fast cache
        location = slow_to_fast[-1]
        f = disk_cacher(location,write_back=True)(f)
        return f
    return hierarchical


#############################################################################
# Setup advanced memoization

ramdisk_location   = '/media/neurotools_ramdisk'
ssd_cache_location = '/ssd_1/mrule'
hdd_cache_location = '/ldisk_1/mrule'

disk_cache_hierarchy = (
    ramdisk_location,
    ssd_cache_location,
    hdd_cache_location)

def purge_ram_cache():
    os.system('rm -rf ' + ramdisk_location + os.sep + CACHE_IDENTIFIER)

def purge_ssd_cache():
    os.system('rm -rf ' + ssd_cache_location + os.sep + CACHE_IDENTIFIER)

# create a ramdisk.
if not os.path.isdir(ramdisk_location):
    print('Initializing public ramdisk at %s'%ramdisk_location)
    os.system('sudo mkdir -p '+ramdisk_location)
    os.system('sudo mount -t tmpfs -o size=16384M tmpfs %s/'%ramdisk_location)
    os.system('sudo chmod -R 777 %s'%ramdisk_location)
else:
    print('existing ramdisk found, will not remount')

# A hierarchical disk cacher
leviathan = hierarchical_cacher(*disk_cache_hierarchy)

# Replace the memoize definition system wide.
# This must happen BEFORE all other imports
# 
old_memoize = neurotools.decorator.memoize
def new_memoize(f):
    return old_memoize(leviathan(f))
memoize = new_memoize
neurotools.decorator.memoize = memoize

def launch_cache_synchronizers():
    # write_back is disabled for the slow disk caches. To make these
    # persistant, we need to run an rsync job to keep them synchronized. 
    for level in range(len(disk_cache_hierarchy)-1):
        source       = disk_cache_hierarchy[level  ] + os.sep + CACHE_IDENTIFIER
        destination  = disk_cache_hierarchy[level+1] + os.sep + CACHE_IDENTIFIER
        destination + CACHE_IDENTIFIER
        # quiet rsync command in update-archive mode
        rsync = "rsync -aqu '%s/' '%s' "%(source,destination)
        # Run the synchronization jobs at idle level
        nice  = "ionice -c3 "
        # loop every 30 seconds at most
        watch = "watch -n60 "
        # silence the output and run in background
        job   = ' 2>1 1>/dev/null &'
        # build command
        command = watch + nice + rsync + job

        #pid = subprocess.Popen(command,creationflags=DETACHED_PROCESS).pid

        print(command)
        os.system(command)
    os.system('reset')

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

