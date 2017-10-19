#!/user/bin/env python 
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


from neurotools.jobs.cache import *

'''
Static initialization routines accompanying `neurotools.jobs.cache`.
These routines were written to set up a caching framework for the Oscar
high performance computing cluster at Brown University, and have not yet
been modified for general use. They still contain hard-coded user-specific
paths, for example.
'''

CACHE_IDENTIFIER ='.__neurotools_cache__'
VERBOSE_CACHING = 0

######################################################################
# Setup advanced memoization

def purge_ram_cache():
    '''
    Deletes the ramdisk cache. USE WITH CAUTION. This depends on the
    global confuration variable `ssd_cache_location` as well as
    `CACHE_IDENTIFIER`.
    
    This will `rm -rf` the entire  `ramdisk_location` and is EXTREMELY
    dangerous. It has been disabled and now raises `NotImplementedError`
    '''
    raise NotImplementedError('cache purging is too dangerous and has been disabled');
    os.system('rm -rf ' + ramdisk_location + os.sep + CACHE_IDENTIFIER)

def purge_ssd_cache():
    '''
    Deletes the SSD drive cache. USE WITH CAUTION. This depends on the
    global confuration variable `ssd_cache_location` as well as
    `CACHE_IDENTIFIER`.
    
    This will `rm -rf` the entire  `ssd_cache_location` and is EXTREMELY
    dangerous. It has been disabled and now raises `NotImplementedError`
    '''
    raise NotImplementedError('cache purging is too dangerous and has been disabled');
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

def reset_ramdisk():
    '''
    This will generate a 500GB ramdisk on debian linux. This allows in RAM 
    inter-process communication using the filesystem metaphore. It should
    be considered dangerous. It runs shell commands that
    require `sudo` privileges.
    '''
    raise NotImplementedError('modifying system ramdisk too dangerous and has been disabled');
    print('Initializing public ramdisk at %s'%ramdisk_location)
    os.system('sudo mkdir -p '+ramdisk_location)
    os.system('sudo umount %s/'%ramdisk_location)
    os.system('sudo mount -t tmpfs -o size=500G tmpfs %s/'%ramdisk_location)
    os.system('sudo chmod -R 777 %s'%ramdisk_location)

def launch_cache_synchronizers():
    '''
    Inter-process communication is mediated via shared caches mapped onto
    the file-system. If a collection of processes are distributed over
    a large filesystem, they may need to share data. 
    
    This solution is very bad and has been depricated. It now raises a
    `NotImplementedError`. 
    
    This solution originally spawned rsync jobs to keep a collection of
    locations in the filesystem synchronized. This is bad for the following
    reasons. Mis-configuration can lead to loss of data. Not all jobs
    may need to share all cache values. It is far better to implement
    some sort of lazy protocol. 
    '''
    raise NotImplementedError(
    'cache synchronization can overwite files; it is too dangerous and has been disabled');
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



# Static initialization code
# This should be run with caution
# TODO: move all static initializers into a more appropriate 
# location

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
    
if not os.path.isdir(ramdisk_location): 
    reset_ramdisk()

'''
Caches can be set up in a hierarchy from fast to slow.
If a cache entry is missing in the fast cache, it can be repopulated
from a slower cache. 

For example, a cache hierarchy might include
- local RAMdisk for inter-process communication between life processes
- local SSD for frequently used intermediate value
- local HDD for larger working datasets
- network filesystem for large database
''' 
disk_cache_hierarchy = (ramdisk_location,ssd_cache_location)

disk_cached       = disk_cacher('.')
leviathan         = hierarchical_cacher(disk_cache_hierarchy,method='npy')
unsafe_disk_cache = hierarchical_cacher(disk_cache_hierarchy,method='npy',allow_mutable_bindings=True)
pickle_cache      = hierarchical_cacher(disk_cache_hierarchy,method='pickle')

# neurotools.jobs.decorator.memoize memoizes within the process memory
# leviathan (because it is large and slow and buggy) memoizes within
# memory, ssd, and possible hdd in a hierarchy. 
# this monkey patches neurotools.jobs.decorator.memoize and replaces it
# with the disk cacher, causing all dependent code to automatically
# implement persistent disk-memoization. 
old_memoize = neurotools.jobs.decorator.memoize
new_memoize = leviathan
memoize     = new_memoize
neurotools.jobs.decorator.memoize = new_memoize

# Testing code
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
