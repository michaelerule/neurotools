#!/user/bin/env python 
# -*- coding: UTF-8 -*-
'''
Static initialization routines accompanying `neurotools.jobs.cache`.
These routines were written to set up a caching framework for the Oscar
high performance computing cluster at Brown University, and have not yet
been modified for general use. They still contain hard-coded user-specific
paths, for example.
'''

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

import os
import sys
IN_SPHINX=False
if 'sphinx' in sys.modules:
    print('Inside Sphinx autodoc; NOT loading scipy and pylab namespaces!')
    IN_SPHINX=True  

if not IN_SPHINX:
    VERBOSE_CACHING = 0

import neurotools
import neurotools.jobs
import neurotools.tools
import neurotools.jobs.cache

######################################################################
# Setup advanced memoization

def purge_ram_cache(CACHE_IDENTIFIER='.__neurotools_cache__'):
    '''
    Deletes the ramdisk cache. USE WITH CAUTION. 
    
    This will `rm -rf` the entire  `ramdisk_location` and is EXTREMELY
    dangerous. It has been disabled and now raises `NotImplementedError`
    '''
    raise NotImplementedError('cache purging is dangerous and has been disabled');
    os.system('rm -rf ' + ramdisk_location + os.sep + CACHE_IDENTIFIER)

def purge_ssd_cache(CACHE_IDENTIFIER ='.__neurotools_cache__'):
    '''
    Deletes the SSD drive cache. USE WITH CAUTION.
    
    This will `rm -rf` the entire  `ssd_cache_location` and is EXTREMELY
    dangerous. It has been disabled and now raises `NotImplementedError`
    '''
    raise NotImplementedError('cache purging is dangerous and has been disabled');
    os.system('rm -rf ' + ssd_cache_location + os.sep + CACHE_IDENTIFIER)

def du(location):
    '''
    Returns the disk usave (`du`) of a file
    
    Parameters
    ----------
    location : string
        Path on filesystem
    
    Returns
    -------
    du : integer
        File size in bytes
    '''
    st = os.stat(location)
    du = st.st_blocks * st.st_blksize
    return du

def reset_ramdisk(force=False):
    '''
    This will create a 500GB ramdisk on debian linux. This allows in RAM 
    inter-process communication using the filesystem metaphore. It should
    be considered dangerous. It runs shell commands that
    require `sudo` privileges. In some cases, these commands may not 
    execute automatically (e.g. if called form a Jupyter or IPython 
    notebook inside a browser). In this case, one must run the commands
    by hand. 
    
    
    Parameters
    ----------
    force : bool
        Modifying the configuration of a ramdisk is risky; The function
        fails with a warning unless force is set to true.
    '''
    global ramdisk_location
    if not force:
        raise NotImplementedError(
            'modifying ramdisk is dangerous; use force=True to force');
    print('Initializing public ramdisk at %s'%ramdisk_location)
    def call(cmd):
        print(cmd)
        os.system(cmd)
    call('sudo mkdir -p '+ramdisk_location)
    call('sudo umount %s/'%ramdisk_location)
    call('sudo mount -t tmpfs -o size=500G tmpfs %s/'%ramdisk_location)
    call('sudo chmod -R 777 %s'%ramdisk_location)
    # 

def launch_cache_synchronizers(CACHE_IDENTIFIER ='.__neurotools_cache__'):
    '''
    Inter-process communication is mediated via shared caches mapped onto
    the file-system. If a collection of processes are distributed over
    a large filesystem, they may need to share data. 
    
    This solution has been depricated. 
    It now raises a `NotImplementedError`. 
    
    This solution originally spawned rsync jobs to keep a collection of
    locations in the filesystem synchronized. This is bad for the following
    reasons. Mis-configuration can lead to loss of data. Not all jobs
    may need to share all cache values. It is far better to implement
    some sort of lazy protocol. 
    '''
    global ramdisk_location,ssd_cache_location,hdd_cache_location
    raise NotImplementedError('cache synchronization can overwite files; it has been disabled');
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


def initialize_caches(ramdisk=None,ssd=None,hdd=None,force=False,
    verbose=False,
    CACHE_IDENTIFIER ='.__neurotools_cache__'):
    '''
    Static cache initialization code
    This should be run with caution
    
    Caches can be set up in a hierarchy from fast to slow.
    If a cache entry is missing in the fast cache, it can be repopulated
    from a slower cache. 

    For example, a cache hierarchy might include
    - local RAMdisk for inter-process communication between life processes
    - local SSD for frequently used intermediate value
    - local HDD for larger working datasets
    - network filesystem for large database
    
    # neurotools.jobs.ndecorator.memoize memoizes within the process memory
    # leviathan (because it is large and slow and buggy) memoizes within
    # memory, ssd, and possible hdd in a hierarchy. 
    # this patches neurotools.jobs.ndecorator.memoize and replaces it
    # with the disk cacher, causing all dependent code to automatically
    # implement persistent disk-memoization. 
    
    This function will need to be called *before* importing other
    libraries.
    
    Other Parameters
    ----------------
    ramdisk: str
        Path to ram disk for caching intermediate results
    ssd: str
        Path to SSD to provide persistent storage (back the ram disk)
    hdd: str
        Optional path to a hard disk; larger but slower storage space. 
    force: boolean, False
        The disk caching framework is still experimental and could lead
        to loss of data if there is a bug (or worse!). By default, 
        this routine and its subroutines will not run unless forced. 
        By requiring the user to set force=true excplitly, we hopefully
        enforce caution when using this functionality. 
        
    Returns
    -------
    
    Example
    -------
    This code was originalled called from a function like this, set up
    for specific configurations in the Truccolo lab
    ::
    
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
            ramdisk_location = ssd_cache_location = hdd_cache_location = None
    '''
    global ramdisk_location, ssd_cache_location, hdd_cache_location
    ramdisk_location,ssd_cache_location,hdd_cache_location = ramdisk,ssd,hdd
    
    if not os.path.isdir(ramdisk_location): 
        reset_ramdisk(force)
    
    hierarchy = (ramdisk,)
    if not ssd is None:
        hierarchy += (ssd,)
    if not hdd is None:
        hierarchy += (hdd,)
    
    # These caches become global attributes and are used for memoization
    neurotools.jobs.initialize_system_cache.disk_cached       =\
         neurotools.jobs.cache.disk_cacher('.',verbose=verbose)
    neurotools.jobs.initialize_system_cache.leviathan         =\
         neurotools.jobs.cache.hierarchical_cacher(\
            hierarchy,
            method='npy',
            verbose=verbose,
            CACHE_IDENTIFIER=CACHE_IDENTIFIER)
    neurotools.jobs.initialize_system_cache.unsafe_disk_cache =\
        neurotools.jobs.cache.hierarchical_cacher(\
            hierarchy,
            method='npy',
            allow_mutable_bindings=True,
            verbose=verbose,
            CACHE_IDENTIFIER=CACHE_IDENTIFIER)
    neurotools.jobs.initialize_system_cache.pickle_cache      =\
        neurotools.jobs.cache.hierarchical_cacher(\
            hierarchy,
            method='pickle',
            verbose=verbose,
            CACHE_IDENTIFIER=CACHE_IDENTIFIER)
    # Replace memoization decorator with disk-cached memoization
    neurotools.jobs.initialize_system_cache.old_memoize =\
         neurotools.jobs.ndecorator.memoize
    neurotools.jobs.initialize_system_cache.new_memoize = leviathan
    neurotools.jobs.initialize_system_cache.memoize     = new_memoize
    neurotools.jobs.ndecorator.memoize = new_memoize

def cache_test():
    '''
    Run a test of the disk cache to see if everything is ok; Called if
    this script is run as main. 
    '''
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

# Testing code
if not IN_SPHINX and (__name__=="__main__" or __name__=='__MAIN__'):
    cache_test()

