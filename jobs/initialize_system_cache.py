#!/user/bin/env python 
# -*- coding: UTF-8 -*-
'''
Static initialization routines accompanying `neurotools_cache`.
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

import os, sys
IN_SPHINX='sphinx' in sys.modules
if IN_SPHINX:
    print('Inside Sphinx autodoc; NOT loading scipy and pylab namespaces!')
else:
    VERBOSE_CACHING = 0
import traceback
import neurotools
import neurotools.jobs
from neurotools.jobs import cache as neurotools_cache

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
    
    This will `rm -rf` the entire  `level2_location` and is EXTREMELY
    dangerous. It has been disabled and now raises `NotImplementedError`
    '''
    raise NotImplementedError('cache purging is dangerous and has been disabled');

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

default_ramdisk_location = os.path.expanduser('~/.neurotools_ramdisk')

def reset_ramdisk(force=False,override_ramdisk_location=None):
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
    global default_ramdisk_location,ramdisk_location
    try:    location_defined = not ramdisk_location is None
    except: location_defined = False
    if not  location_defined:
        # Ramisk location might not be initialized
        ramdisk_location =\
            default_ramdisk_location\
            if override_ramdisk_location is None\
            else override_ramdisk_location
    if not force:
        raise NotImplementedError(
            'modifying ramdisk is dangerous; use force=True to force');
    print('Initializing public ramdisk at %s'%ramdisk_location)
    commands = [
        'sudo mkdir -p '+ramdisk_location,
        'sudo umount %s/'%ramdisk_location,
        'sudo mount -t tmpfs -o size=500G tmpfs %s/'%ramdisk_location,
        'sudo chmod -R 777 %s'%ramdisk_location
    ]
    print('To initialize a ramdisk on linux, run these commands:')
    print('\t'+'\n\t'.join(commands))
    print('... attempting to call these commands from python')
    print('(this will not work in jupyter/ipython notebook)')
    def call(cmd):
        print(cmd)
        os.system(cmd)
    for cmd in commands:
        call(cmd)
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
    global ramdisk_location,level2_location,level3_location
    raise NotImplementedError('cache synchronization overwites files; this is dangerous and it has been disabled');
    # write_back is disabled for the slow disk caches. To make these
    # persistant, we need to run an rsync job to keep them synchronized.
    disk_cache_hierarchy = (ramdisk_location,level2_location,level3_location)
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


def initialize_caches(level1=default_ramdisk_location,level2=None,level3=None,force=False,
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
    # unsafe_disk_cache memoizes within
    # memory, ssd, and possible hdd in a hierarchy. 
    # this patches neurotools.jobs.ndecorator.memoize and replaces it
    # with the disk cacher, causing all dependent code to automatically
    # implement persistent disk-memoization. 
    
    This function will need to be called *before* importing other
    libraries.
    
    Other Parameters
    ----------------
    level1: str
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
            level1_location   = '/media/neurotools_level1'
            level2_location = '/ssd_1/mrule'
            level3_location = '/ldisk_1/mrule'
        elif myhost in ('basecamp',):
            level1_location   = '/media/neurotools_level1'
            level2_location = '/home/mrule'
        elif myhost in ('RobotFortress','petra'):
            level1_location   = '/Users/mrule/neurotools_level1'
            level2_location = '/Users/mrule'
        else:
            print('New System. Cache Locations will need configuring.')
            level1_location = level2_location = level3_location = None
    '''
    global level1_location, level2_location, level3_location
    
    level1_location,level2_location,level3_location = level1,level2,level3
    
    if not os.path.isdir(level1_location): 
        raise ValueError('Level 1 location should be an existing directory')
    #    reset_level1(force)
    
    # Add additional cache levels if they are specified 
    hierarchy = (level1,)
    if not level2 is None: hierarchy += (level2,)
    if not level3 is None: hierarchy += (level3,)
    
    # Define the disk-caching memoization decorator
    neurotools.jobs.initialize_system_cache.unsafe_disk_cache =\
        neurotools_cache.hierarchical_cacher(\
            hierarchy,
            method='npy',
            allow_mutable_bindings=True,
            verbose=verbose,
            CACHE_IDENTIFIER=CACHE_IDENTIFIER)

    # Replace in-memory memoization with disk-cached memoization
    neurotools.jobs.initialize_system_cache.old_memoize = neurotools.jobs.ndecorator.memoize
    neurotools.jobs.initialize_system_cache.new_memoize = unsafe_disk_cache
    neurotools.jobs.initialize_system_cache.memoize     = new_memoize
    neurotools.jobs.ndecorator.memoize                  = new_memoize

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
        print('\tState is:',a,b,c,d,e,f,g)
        return [1,1,1,1]

    h = new_memoize(example_function)
    
    print('Testing argument siganture encoding')
    f   = example_function
    sig = ((('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')), None)
    for mode in ['repr','json','human']: #'pickle',
        print('\n',mode)
        print('\t',neurotools.jobs.cache.function_signature(f))
        print('\t',sig)
        try:
            fn  = neurotools.jobs.cache.signature_to_file_string(f,sig,mode=mode)
            print('\t',fn)
            s2  = neurotools.jobs.cache.file_string_to_signature(fn,mode=mode)
            print('\t',s2)
        except ValueError:
            traceback.print_exc()

    # test hierarchical cache
    print('Testing hybrid caching')
    print('Caution this is experimental')
    fn = unsafe_disk_cache(example_function)
    print('Testing unsafe_disk_cache ',fn.__name__)
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

