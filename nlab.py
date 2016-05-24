#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division

# Start migrating to print-as-a-function 
# from __future__ import print_function

import os, sys
import pickle
import random
import traceback
import warnings

from   collections       import *
from   itertools         import *
from   os.path           import *
from   multiprocessing   import Process, Pipe, cpu_count, Pool

import scipy
import scipy.optimize

try:
    from   sklearn.metrics   import roc_auc_score,roc_curve,auc
except Exception, e:
    traceback.print_exc()
    print 'Importing sklearn failed, ROC and AUC will be missing'

from   scipy.stats       import wilcoxon
from   scipy.signal      import *
from   scipy.optimize    import leastsq
from   scipy.interpolate import *
from   scipy.io          import *
from   scipy.signal      import butter,filtfilt,lfilter

#from   pylab             import *
#set_printoptions(precision=2)

'''
# messes with backend -- don't run this
# except sometimes it may be necessary on OSX? unclear.
import matplotlib
import matplotlib.pyplot as plt
import sys
print matplotlib.pyplot.get_backend()
modules = []
for module in sys.modules:
    if module.startswith('matplotlib'):
        modules.append(module)
for module in modules:
    sys.modules.pop(module)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print matplotlib.pyplot.get_backend() 
'''


print 'Loading nlab namespace'
from neurotools.spike               import *
from neurotools.tools               import *
from neurotools.ntime               import *
from neurotools.functions           import *
from neurotools.namespace           import *
from neurotools.color               import *
from neurotools.plot                import *

from neurotools.lif                 import *

from neurotools.spatial.array               import *
from neurotools.spatial.distance            import *
from neurotools.spatial.fftzeros            import *
from neurotools.spatial.spatialPSD          import *
from neurotools.spatial.phase              import *

from neurotools.stats.density             import *
from neurotools.stats.entropy             import *
from neurotools.stats.GLMFit              import *
from neurotools.stats.glm                 import *
from neurotools.stats.history_basis       import *
from neurotools.stats.kent_reimann        import *
from neurotools.stats.stats               import *
from neurotools.stats.magickernel         import *
from neurotools.stats.modefind            import *
from neurotools.stats.regressions         import *
from neurotools.stats.vonmises            import *

from neurotools.signal.linenoise           import *
from neurotools.signal.morlet_coherence    import *
from neurotools.signal.morlet              import *
from neurotools.signal.multitaper          import *
from neurotools.signal.ppc                 import *
from neurotools.signal.savitskygolay       import *
from neurotools.signal.signal              import *
from neurotools.signal.dct                 import *
from neurotools.signal.coherence           import *
from neurotools.signal.conv                import *

from neurotools.jobs.parallel            import *
from neurotools.jobs.decorator           import *

print 'Restoring pylab namespace (in event of accidental shadowing of pylab functions)'
# from scipy.stats    import *
from neurotools.getfftw             import *
# seems like too many imports ruining pylab context
# try to fix it up a bit
# from pylab    import *

# suppress verbose warningmessages
nowarn()

from numpy.core.multiarray import concatenate as cat





