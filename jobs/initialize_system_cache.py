#!/user/bin/env python 
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
# disk_cache_hierarchy = (ramdisk_location,ssd_cache_location)

# disk_cached       = disk_cacher('.')
# leviathan         = hierarchical_cacher(disk_cache_hierarchy,method='npy')
# unsafe_disk_cache = hierarchical_cacher(disk_cache_hierarchy,method='npy',allow_mutable_bindings=True)
# pickle_cache      = hierarchical_cacher(disk_cache_hierarchy,method='pickle')

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
