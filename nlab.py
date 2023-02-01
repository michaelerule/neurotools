#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
This is a wrapper module that imports pylab as well
as commonly used routines from neurotools. Try

    from nlab import *

'''

from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

from neurotools.util.system import *

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

from neurotools.spikes                   import *
from neurotools.spikes.sta               import *
from neurotools.spikes.waveform          import *
from neurotools.util.tools               import *
from neurotools.util.time                import *
from neurotools.util.string              import *
from neurotools.util.functions           import *
from neurotools.linalg.operators         import *
from neurotools.graphics.color           import *
from neurotools.graphics.colormaps       import *
from neurotools.graphics.plot            import *
from neurotools.linalg.matrix            import *

from neurotools.spatial.dct              import *
from neurotools.spatial.array            import *
from neurotools.spatial.distance         import *
from neurotools.spatial.fftzeros         import *
from neurotools.spatial.phase            import *
from neurotools.spatial.spiking          import *
from neurotools.spatial.kernels          import *

import neurotools.stats
from neurotools.stats                    import *
from neurotools.stats.density            import *
from neurotools.stats.distributions      import *
from neurotools.stats.mixtures           import *
from neurotools.stats.information        import *
from neurotools.stats.glm                import *
from neurotools.stats.hmm                import *
from neurotools.stats.modefind           import *
from neurotools.stats.regressions        import *
from neurotools.signal.morlet            import *
#from neurotools.signal.phase             import *
from neurotools.signal.savitskygolay     import *

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

from neurotools.util.getfftw                 import *

try:
    from neurotools.util.hdfmat                  import *
except ModuleNotFoundError:
    print('ModuleNotFoundError: No module named \'h5py\'; please install this to use neurotools.hdfmat')

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

from neurotools.graphics.color import BLACK
mpl.rcParams['axes.prop_cycle'] = cycler(color=[BLACK,RUST,TURQUOISE,OCHRE,AZURE])



