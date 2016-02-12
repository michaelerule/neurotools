
import os, sys
import pickle
import random
import traceback

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
from neurotools.amplitudetransients import *
from neurotools.dct                 import *
from neurotools.array               import *
from neurotools.coherence           import *
from neurotools.conv                import *
from neurotools.density             import *
from neurotools.distance            import *
from neurotools.entropy             import *
from neurotools.fftzeros            import *
from neurotools.GLMFit              import *
from neurotools.glm                 import *
from neurotools.history_basis       import *
from neurotools.kent_reimann        import *
from neurotools.lif                 import *
from neurotools.linenoise           import *
from neurotools.magickernel         import *
from neurotools.modefind            import *
from neurotools.morlet_coherence    import *
from neurotools.morlet              import *
from neurotools.multitaper          import *
from neurotools.namespace           import *
from neurotools.parallel            import *
from neurotools.color               import *
from neurotools.phase               import *
from neurotools.plot                import *
from neurotools.ppc                 import *
from neurotools.regressions         import *
from neurotools.savitskygolay       import *
from neurotools.signal              import *
from neurotools.spatialPSD          import *
from neurotools.spike               import *
from neurotools.stats               import *
from neurotools.ntime               import *
from neurotools.tools               import *
from neurotools.vonmises            import *
from neurotools.time                import *
from neurotools.functions           import *

print 'Restoring pylab namespace (in event of accidental shadowing of pylab functions)'
# from scipy.stats    import *
from neurotools.getfftw             import *
# seems like too many imports ruining pylab context
# try to fix it up a bit
# from pylab    import *

# suppress verbose warningmessages
nowarn()

from numpy.core.multiarray import concatenate as cat





