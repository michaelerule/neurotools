## Computational neuroscience python routines. 

This is a collection of python modules developed for exploratory analysis and statistical modeling in computational neuroscience. Most of the code and documentation is of "research" (i.e. poor) quality and unstable, but some of the modules are stable and useful. The (very much under construction) documentation [can be found on github pages](http://michaelerule.github.io/neurotools/_build/html/index.html).

This project depends on
 - [numpy](http://www.numpy.org/)
 - [scipy](https://www.scipy.org/)
 - [statsmodels](http://www.statsmodels.org/stable/index.html)
 - [pandas](http://pandas.pydata.org/)
 - [nitime](http://nipy.org/nitime/)
 - [spectrum](https://pyspectrum.readthedocs.io/en/latest/install.html) 

Numpy, scipy, statsmodels, and pandas are probably provided with most scientific python distributions. You may need into install [nipy](http://nipy.org/nitime/) and [spectrum](https://pyspectrum.readthedocs.io/en/latest/install.html) manually. They are available via pip. 

Some modules have extra dependencies, and will complain accordingly if they are missing when you try to import them. Optional dependences include: 
- sklearn
- pyfftw
- pyopencl
- pycuda
- pygame
- h5py

Unless otherwise specified, media, text, and rendered outputs are licensed under the [Creative Commons Attribution Share Alike 4.0 license](https://choosealicense.com/licenses/cc-by-sa-4.0/) (CC BY-SA 4.0). Source code is licensed under the [GNU General Public License version 3.0](https://www.gnu.org/copyleft/gpl.html) (GPLv3). The CC BY-SA 4.0 is [one-way compatible](https://creativecommons.org/compatiblelicenses) with the GPLv3 license. 
This license does not apply to the project as a whole, but only to those modules or functions for which an alternative license is not otherwise specified. This project aggregates several functions published informally (e.g. as stackoverflow answers or via gist). Routines taken from elsewhere are cited as such, with a link to the original source provided.
