'''
Some notes

#############################################################################
# To build the documentation
#http://sphinx-doc.org/tutorial.html

$ (sudo?) pip install Sphinx

# https://codeandchaos.wordpress.com/2012/07/30/sphinx-autodoc-tutorial-for-dummies/
# the -F option invokes sphinx-quickstart and folllowing

# call quickstart, there are many options
$ sphinx-quickstart

.
y
_
neurotools
mrule
0
0
en
.rst
index
n
y
y
n
y
y
n
y
y
y
y
y

The package is outside the python package path so sphynx won'tbe able to 
find it for autodoc unless we add these lines to config.py

>>> from os.path import expanduser, abspatch
>>> sys.path.insert(0,abspath(expanduser('~/Dropbox/bin')))

# Then run 
$ sphinx-apidoc -fe -o ./source .
$ make clean
$ make html

# math is pretty slow, maybe try PNGmath instead of mathjax?
# something is broken with the table of contents
# Let's try this
# http://sphinxcontrib-fulltoc.readthedocs.org/en/latest/install.html

pip install sphinxcontrib-fulltoc

# conf.py
...
extensions = ['sphinxcontrib.fulltoc']

OK so the fulltoc package doesn't install correctly because absolutely 
nothing ever works I guess.

Try it from github?

https://github.com/dreamhost/sphinxcontrib-fulltoc/archive/master.zip

NOO ok you may need to use the -U flag on pip for "upstream" to get
the build dependencies.

and then there are bugs. Some templates crash. Others work, though.

'''


#import os
#from os.path import expanduser
#ntools = expanduser('~/Dropbox/bin/neurotools')
# todo: way to get from * import * ?
#modules = [f[:-3] for f in os.listdir(ntools) if f[-3:]=='.py']
#__all__ = modules
#print '__all__=',__all__
"""
__all__ = ['multitapertools',
 'glm',
 'entropy',
 'coherence',
 'convtools',
 'namespace',
 'phasetools',
 'distancetools',
 'history_basis',
 'spiketools',
 'savitskygolay',
 'morlet_coherence',
 'getfftw',
 'fftzeros',
 'modefind',
 'ppc',
 'GLMFit',
 'huetools',
 'kent_reimann',
 'densitytools',
 'liftools',
 'magickernel',
 'partools',
 'regressions',
 'morlet',
 'nsignaltools',
 'vonmises',
 'tools',
 'arraytools',
 'stattools',
 'amplitudetransients',
 'spatialPSD',
 'linenoisetools',
 'parula',
 'plottools',
 'timetools',
 'functions']
"""
 
 
 
    
    
    
    
 
