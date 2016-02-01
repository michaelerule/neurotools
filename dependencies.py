'''
Dependency check for neurotools. 
'''

# Gather a list of builtins if possible. These are implemented as part
# of the python interpreter and should always exist.
try:
    import sys
    builtins = sys.builtin_module_names
    python_version = sys.version.split('\n')[0].split()[0]
except:
    print 'Somehow the sys package is missing'
    print 'Very funny, are we even in a python interpreter?'
    builtins = []
    python_version = 'UNKNOWN'

try:
    import os.path
except:
    print 'Somehow the os.path package is missing'
    print 'This is part of the standard library and it should be here'

try:
    import glob
except:
    print 'Somehow the glob package is missing'
    print 'This is part of the standard library and it should be here'



# Try to inspect the installed python directory to see what standard
# library modules are available. Again, these should never be missing,
# but may eventually change over time. Also, their version number should
# correspond to the Python version.
try:
    stdlib = []
    for path in glob.glob(sys.prefix + "/lib/python%d.%d" % (sys.version_info[0:2]) + "/*.py"):
        stdlib.append(os.path.basename(path)[:-3])
except:
    print 'I tried to inspect the standard library installation but failed'
    print 'This is alarming. Nonstandard installation or environment?'
    stdlib = []



# Let's hope we can at least find the package for inspecting packages!
try:
    import pkg_resources
except:
    print 'Somehow the pkg_resources package is missing'
    print 'We need this to robustly check versions for compatibility'
    print "I'll do my best without it..."



# Pip may come in handy, depending on whether or not you use pip to manage
# python packages.
installed_via_pip = {}
try:
    import pip
    for package in pip.get_installed_distributions():
        installed_via_pip[package.key]=package
except:
    print 'The pip package is missing.'
    print 'Pip is used to manage and inspect installed packages.'
    print "Python packages may also be installed manually, via easy_install, or via various system package managers, so it's not essential."
    print "Please see https://pip.pypa.io/en/stable/installing/ to install pip"
    print "Alternatively, an installer script should be available at https://bootstrap.pypa.io/get-pip.py"
    print "Note that pip does not work with OS X 10.8 or older"
    # print "Would you like me to try to install pip for you?"



# list 
DEPENDENCIES = [
 # modules built in to the python interpreter. should always exist
 ('time', '2.7.6'),
 ('itertools', '2.7.6'),
 # standard library modules. should exist in standard installations
 ('collections', '2.7.6'),
 ('inspect', '2.7.6'),
 ('functools', '2.7.6'),
 ('os', '2.7.6'),
 ('pickle', '2.7.6'),
 ('random', '2.7.6'),
 ('re', '2.2.1'),
 ('shlex', '2.7.6'),
 ('traceback', '2.7.6'),
 ('types', '2.7.6'),
 # custom libraries -- you may have to install these by hand.
 ('decorator', '3.4.0'),        # optional, in PYPI
 ('matplotlib', '1.3.1'),       # required, in PYPI
 ('nitime', '0.5'),             # required, in PYPI
 ('statsmodels', '0.6.1'),      # required, in PYPI
 ('multiprocessing', '0.70a1'), # optional, can use single-threaded fallback, in PYPI
 ('pyfftw', '0.9.2'),           # optional, can use numpy fallback, in PYPI
 ('spectrum', '0.6.0'),         # required, in PYPI
 ('sklearn', '0.15.2'),         # required, in PYPI
 # Numpy and scipy seem to not reliably install over pip/easy_install
 # Possibly due to missing build dependencies? 
 # These will just need to be handled as a special case.
 # http://www.scipy.org/install.html
 ('numpy', '1.9.2'),
 ('scipy', '0.16.0'),
]

intalled_summary = []
missing = []

for entry in DEPENDENCIES:
    if len(entry)==2:
        package,version = entry
        note = ''
    elif len(entry)==3:
        package,version,note = entry
    
    print 'depends on',package
    if package in builtins:
        print '\tthis is a builtin, it should never be missing'
    elif package in stdlib:
        print '\tthis is part of the standard library'
    try:
        mod = __import__(package)
        print '\tis installed'
    except:
        print '\timport failed, %s may not be installed or python path may not me correctly configured'%package
        missing.append(package)
        continue # move on to next dependency

    # due to potential weirdness that may arise with python 
    # environments, it's not clear that the imported version will 
    # always match the one reported via pip. For this reason, we
    # actually do the import and try to read the version name from 
    # the package itself. This doesn't always work, so we use pip
    # as a fallback
    potential_version_variable_names = ['__version__','__VERSION__','VERSION','version','version_info']
    loaded_version = None
    if loaded_version is None:
        for vname in potential_version_variable_names:
            if vname in mod.__dict__:
                # try to find version information, 
                # just hope and pray it's a string if it exists
                # take only the first line if multiple lines exist
                loaded_version = mod.__dict__[vname]
                break
    if loaded_version is None:
        try:
            loaded_version = pkg_resources.get_distribution(package).version
        except:
            pass
    if loaded_version is None:
        if package in installed_via_pip:
            p = installed_via_pip[package]
            if p.has_version:
                loaded_version = p.version
    
    if package in builtins:
        # default to reporting the python version for builtins
        if loaded_version is None:
            loaded_version = python_version
        # if a version number was reported, sanity check that it matches
        elif loaded_version != python_version:
            print '\tA builtin is reporting a version number (unusual)'
            print '\t\tthat differs from the Python version.'
            print '\t\tThis is unexpected'
            print '\t\tmodule reported',loaded_version
            print '\t\tPython version is',python_version
            loaded_version = python_version
                
    if package in stdlib:
        # default to reporting the python version for the standard library
        if loaded_version is None:
            loaded_version = python_version
        # if a version number was reported, sanity check that it matches
        elif loaded_version != python_version:
            print '\tA standard library module is reporting a version number that differs from the Python version.'
            print '\t\tmodule reported',loaded_version
            print '\t\tPython version is',python_version
            loaded_version = python_version
                
    if loaded_version is None:
        print '\tNo version information found.'
    else:
        loaded_version = loaded_version.split('\n')[0]
        print '\tVersion',loaded_version 
        if loaded_version != version:
            print '\tThe loaded version differs from the dependency version'
            print '\t\tThe loaded version is',loaded_version
            print '\t\tThe required version is',version
            print '\t\tThis is usually OK, but it may cause instability and surprising crashes'

    intalled_summary.append((package,loaded_version))


def ask(msg,default=True):
    no_answer = True
    answer = None
    while no_answer:
        var = raw_input(msg+(' [Y/n]?' if default else ' [y/N]?'))
        if len(var)<=0:
            answer = default
            break
        var = var[0].lower()
        if var in 'yn':
            answer = var=='y'
            break
        if default: print "please type n(o), or press enter for y(es)"
        else:       print "please type y(es), or press enter for n(o)"
    return answer    


# check to see which setup tools are available
try:
    import setuptools
    from setuptools.command import easy_install
except:
    easy_install = None
try:
    import pip
except:
    pip = None

if easy_install==None and pip==None:
    print 'Neither pip nor easy_install is available, so I will not try to install missing packages.'
    print 'Please install the following packages manually'
    print '\t'+'\n\t'.join(missing)
else:
    # we can try to install things automatically
    for package in missing:
        if package in builtins:
            print 'This is a bug, %s is a builtin, it is not missing'%package
            continue
        if package in stdlib:
            print 'This is a bug, %s is in the standard library, it is not missing'%package
            print "This might happen if you've configured a very unusual environment, but I'm not going to try to fix it"
            continue
        if package in ['numpy','scipy']:
            print 'Package %s is missing, but automatic may be problematic.'
            print 'Please search online and follow installation instructions for your platform'
            continue
        if ask('Package %s is missing, should I try to install it'%package):
            if not pip==None:
                pip.main(['install', package])
            else:
                print 'pip is missing, I will try easy_install'
                easy_install.main( ["-U",package] )

    


