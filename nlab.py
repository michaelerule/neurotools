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
from neurotools.text                     import *
from neurotools.functions                import *
from neurotools.linalg.operators         import *
from neurotools.graphics.color           import *
from neurotools.graphics.colormaps       import *
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

import neurotools.stats
from neurotools.stats                    import *
from neurotools.stats.density            import *
from neurotools.stats.distributions      import *
from neurotools.stats.mixtures           import *
from neurotools.stats.entropy            import *
from neurotools.stats.GLMFit             import *
from neurotools.stats.glm                import *
from neurotools.stats.hmm                import *
from neurotools.stats.gmm                import *
from neurotools.stats.mvg                import *
from neurotools.stats.history_basis      import *
from neurotools.stats.kent_reimann       import *
from neurotools.stats.modefind           import *
from neurotools.stats.regressions        import *
from neurotools.stats.circular           import *

from neurotools.signal.linenoise         import *
from neurotools.signal.morlet_coherence  import *
from neurotools.signal.morlet            import *

# Depends on the spectrum package and will not import if this is missing
try:
    from neurotools.signal.multitaper        import *
except ImportError:
    print('Skipping the neurotools.signal.multitaper module')

# Depends on the spectrum package and will not import if this is missing
try:
    from neurotools.signal.ppc               import *
except ImportError:
    print('Skipping the neurotools.signal.ppc module')

from neurotools.signal.savitskygolay     import *
from neurotools.signal                   import *

# Depends on the nitime package and will not import if this is missing
try:
    from neurotools.signal.coherence         import *
except ImportError:
    print('Skipping the neurotools.signal.coherence module')

from neurotools.signal.conv              import *

# Sometimes this fails?
try:
    from neurotools.jobs.parallel            import *
    from neurotools.jobs.ndecorator          import *
except ImportError:
    print('Skipping the neurotools.jobs package')

from neurotools.getfftw                  import *

from neurotools.hdfmat                   import *

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
from matplotlib.pyplot import *
import scipy
from scipy         import *
from scipy.special import *
from scipy.linalg  import *
from numpy.random import *
from numpy import *
from pylab import *

# Mess with matplotlib
rcParams['figure.dpi']=120
plt.rcParams['image.cmap'] = 'parula'
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=[BLACK,RUST,TURQUOISE,OCHRE,AZURE])



