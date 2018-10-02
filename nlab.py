#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

print( 'Loading nlab namespace')

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
except Exception as e:
    print('could not find sklearn; ROC and AUC will be missing')

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

from neurotools.spikes.spikes            import *
from neurotools.spikes.waveform          import *
from neurotools.tools                    import *
from neurotools.functions                import *
from neurotools.linalg.operators         import *
from neurotools.graphics.color           import *
from neurotools.graphics.plot            import *
from neurotools.linalg.matrix            import *

from neurotools.models.lif               import *
from neurotools.models.izh               import *

from neurotools.spatial.dct              import *
from neurotools.spatial.array            import *
from neurotools.spatial.distance         import *
from neurotools.spatial.fftzeros         import *
from neurotools.spatial.spatialPSD       import *
from neurotools.spatial.phase            import *
from neurotools.spatial.spiking          import *
from neurotools.spatial.kernels          import *

from neurotools.stats.density            import *
from neurotools.stats.distributions      import *
from neurotools.stats.mixtures           import *
from neurotools.stats.entropy            import *
from neurotools.stats.GLMFit             import *
from neurotools.stats.glm                import *
from neurotools.stats.hmm                import *
from neurotools.stats.gmm                import *
from neurotools.stats.history_basis      import *
from neurotools.stats.kent_reimann       import *
from neurotools.stats.stats              import *
from neurotools.stats.modefind           import *
from neurotools.stats.regressions        import *
from neurotools.stats.circular           import *

from neurotools.signal.linenoise         import *
from neurotools.signal.morlet_coherence  import *
from neurotools.signal.morlet            import *
from neurotools.signal.multitaper        import *
from neurotools.signal.ppc               import *
from neurotools.signal.savitskygolay     import *
from neurotools.signal                   import *
from neurotools.signal.coherence         import *
from neurotools.signal.conv              import *

from neurotools.jobs.parallel            import *
from neurotools.jobs.ndecorator          import *

from neurotools.getfftw                  import *

# suppress verbose warning messages
nowarn()

from numpy.core.multiarray import concatenate as cat

try:
    import h5py
except:
    print('could not locate h5py; support for hdf5 files missing')
    h5py = None

# Last but not least 
from pylab import *   
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
