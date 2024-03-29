#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
import sys, os


html_logo = 'logo1.svg'

# custom.css is inside one of the html_static_path folders (e.g. _static)
html_css_files = ["custom.css"]

# -- General configuration ------------------------------------------------
# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '4.0'
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    ]

# special code to handle different versions of sphinx gracefully
mathjax_path="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
def addpath(p):
    p = os.path.abspath(p)
    print('Adding path',p)
    sys.path.insert(0,p)
addpath('../')
addpath('../../')

# Awful hack?
import neurotools
from neurotools import *
from neurotools import stats
import nlab
from nlab import *
import neurotools.signal as signal
import neurotools.stats as stats
import neurotools.linalg as linalg
import neurotools.jobs as jobs
import neurotools.spatial as spatial
import neurotools.graphics as graphics

import imp, traceback

allregistered = []

for retry in range(10):
    for dp,dn,fns in os.walk('../'):
        p = os.path.abspath(dp)
        if 'docs' in p: continue
        if 'source' in p: continue
        if '__init__.py' in [f.lower() for f in fns]:
            # directory is a module
            mn = p.split(os.sep)[-1]
            if mn in globals(): continue
            print('Probably a package: ',p,mn)
            try:
                globals()[mn] = imp.load_source(mn,p)
                print('loaded.')
            except:
                p2 = p+os.sep+'__init__.py'
                try:
                    globals()[mn] = imp.load_source(mn,p2)
                    print('loaded.')
                except:
                    print('MODUlE LOADING FAILED WITH ERROR')
                    traceback.print_exc()
                    continue
            allregistered.append(mn)
            parent = os.path.abspath(os.path.join(p, os.pardir))
            siblings = os.listdir(parent)
            if not '__init__.py' in [f.lower() for f in siblings]:
                print('Modules %s DOES NOT SEEM TO HAVE A PACKAGE?'%mn)
                continue
            try:
                # directory is a module
                pn = parent.split(os.sep)[-1]
                if not pn in globals():
                    print('PARENT MODULE %s NOT LOADED'%pn)
                    continue
                setattr(globals()[pn],mn,globals()[mn])
                print('Registered as %s.%s'%(pn,mn))
                # somehow... build up path?
                allregistered.append('%s.%s'%(pn,mn))
            except:
                print('COULD NOT REGISTER WITH PARENT')
                traceback.print_exc()
                continue
    for dp,dn,fns in os.walk('../'):
        p = os.path.abspath(dp)
        if 'docs' in p: continue
        if 'source' in p: continue
        for fn in fns:
            if not fn[-3:].lower()=='.py': continue
            if '__init__' in fn.lower(): continue
            mn = fn.split('.')[0]
            if mn in globals(): continue
            fp = p+os.sep+fn
            print('Probably a module: ',fp,mn)
            try:
                globals()[mn] = imp.load_source(mn,fp)
                print('loaded',mn)
            except:
                print('MODUlE LOADING FAILED WITH ERROR')
                traceback.print_exc()
            allregistered.append(mn)
            if not '__init__.py' in [f.lower() for f in fns]:
                print('Modules %s DOES NOT SEEM TO HAVE A PACKAGE?'%mn)
                continue
            try:
                # directory is a module
                pn = p.split(os.sep)[-1]
                if not pn in globals():
                    print('PARENT MODULE %s NOT LOADED'%pn)
                    continue
                setattr(globals()[pn],mn,globals()[mn])
                print('Registered as %s.%s'%(pn,mn))
                # somehow... build up path?
                allregistered.append('%s.%s'%(pn,mn))
            except:
                print('COULD NOT REGISTER WITH PARENT')
                traceback.print_exc()

print('Imported these packages:\n\t'+'\n\t'.join(sorted(allregistered)))


# due to potential weirdness that may arise with python 
# environments, it's not clear that the imported version will 
# always match the one reported via pip. For this reason, we
# actually do the import and try to read the version name from 
# the package itself. This doesn't always work, so we use pip
# as a fallback
'''
mod = __import__('sphinx')
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
if loaded_version is None:
    sys.stdout.write('\tNo sphinx version information found.')
else:
    loaded_version = loaded_version.split('\n')[0]
    loaded_version = tuple(map(int,loaded_version.split('.')))
    sys.stdout.write('\tSphinx version '+str(loaded_version)+'\n')
    if loaded_version>=(1,4):
        extensions += [
        'sphinxcontrib.fulltoc',
        'sphinx.ext.githubpages',]
        sys.stdout.write('\tAdding github pages module\n')
    else:
        sys.stdout.write('\tPlease update Sphinx to use the github pages module\n')
'''

# fix some sphinx warnings?
#napoleon_use_param = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Neurotools'
copyright = u'2017, M Rule'
author = u'M Rule'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u'2'
# The full version, including alpha/beta/rc tags.
release = u'2'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','extract_doc.py']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# https://stackoverflow.com/questions/27669376/show-entire-toctree-in-read-the-docs-sidebar
html_theme_options = {
    'navigation_depth': 3,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'Neurotools'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'Neurotools.tex', u'Neurotools Documentation',
     u'M Rule', 'manual'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'Neurotools', u'Neurotools Documentation',
     [author], 1)
]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Neurotools', u'Neurotools Documentation',
     author, 'Neurotools', 'One line description of project.',
     'Miscellaneous'),
]

# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']



####### FURHTHER CUSTOMIZATION RTD ########

# https://stackoverflow.com/questions/18969093/how-to-include-the-toctree-in-the-sidebar-of-each-page
# html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'], }

# https://stackoverflow.com/questions/20939598/enabling-sidebar-on-python-sphinx-documentation-based-on-a-sphinx-bootstrap-them
html_sidebars = {'**': ['localtoc.html', 'sourcelink.html', 'searchbox.html']}


extensions += ['sphinx.ext.autosummary',]
autodoc_default_flags  = ['members']
autosummary_gerenerate = True
exclude_patterns       = ['_auto/*']
autodoc_member_order   = 'bysource'
autodoc_mock_imports = ["pylab", 'numpy', 'scipy', 'numpy.random', 'pylab' ]

# Patch duplicate errors
# https://github.com/sphinx-doc/sphinx/issues/3866

from sphinx.domains.python import PythonDomain

class PatchedPythonDomain(PythonDomain):
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        if 'refspecific' in node:
            del node['refspecific']
        return super(PatchedPythonDomain, self).resolve_xref(
            env, fromdocname, builder, typ, target, node, contnode)
    
def setup(sphinx):
    #sphinx.override_domain(PatchedPythonDomain)
    pass

# Get notebook support
# https://nbsphinx.readthedocs.io/en/0.2.17/installation.html
# python3 -m pip install nbsphinx --user

