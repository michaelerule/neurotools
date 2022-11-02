.. Neurotools documentation master file, created by
   sphinx-quickstart on Sun Jul  9 15:44:41 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Neurotools
==========

Python modules for statistical modeling in computational neuroscience. Most of the code and documentation is of "research" (i.e. poor) quality and unstable, but some of the modules are stable and useful. (Refer to specific modules for further documentation.)

This project depends on `numpy
<http://www.numpy.org/>`_, `scipy
<https://www.scipy.org/>`_, `statsmodels
<http://www.statsmodels.org/stable/index.html>`_, `pandas
<http://pandas.pydata.org/>`_, and `nipy
<http://nipy.org/nitime/>`_. This project is not to be confused with the (much better organized) `Neurotools project for neural simulation
<http://neuralensemble.org/NeuroTools/>`_. For the most part the routines here do not duplicate the functionality of these packages. There is a focus on spatiotemporal modeling of multi-electrode array datasets, stochastic models of population dynamics, and some routines to explore population synchrony. There are also several playful or exploratory modules that probably aren't all that useful but are available nevertheless. The project source can be browsed or downloaded from `github
<https://github.com/michaelerule/neurotools>`_.

This project has not been reviewed or prepared for use by the general public. When not otherwise specified this project and associated content is licensed under the `Creative Commons Attribution Non Commercial Share Alike 3.0 license
<https://creativecommons.org/licenses/by-nc-sa/3.0/>`_ `(full license)
<https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode>`_. 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
   :maxdepth: 1
   :caption: Subpackages:

   signal <neurotools.signal>
   stats <neurotools.stats>
   spatial <neurotools.spatial>
   spikes <neurotools.spikes>
   graphics<neurotools.graphics>
   linalg <neurotools.linalg>
   jobs <neurotools.jobs>
   util <neurotools.jobs>
   
