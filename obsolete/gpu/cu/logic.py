#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
This module contains GPU wrappers to perform boolean logic in parallel.
These functions very much need work, since they basically use 
Iverson's convention, but with floats, such that 0.0f = false, 1.0f=true.
Nevertheless I have found these little functions quite useful in their
present form in some circumstances
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function


from neurotools.obsolete.gpu.cu.function import *

gpult    = lambda x:gpumap("$<%f?1:0"%x)
'''lambda x:gpumap("$<%f?1:0"%x)'''
gpugt    = lambda x:gpumap("$>%f?1:0"%x)
'''lambda x:gpumap("$>%f?1:0"%x)'''
gpueq    = lambda x:gpumap("$==%f?1:0"%x)
'''lambda x:gpumap("$>%f?1:0"%x)'''
gpuneq   = lambda x:gpumap("$!=%f?1:0"%x)
'''lambda x:gpumap("$!=%f?1:0"%x)'''
gpulteq  = lambda x:gpumap("$<=%f?1:0"%x)
'''lambda x:gpumap("$<=%f?1:0"%x)'''
gpugteq  = lambda x:gpumap("$>=%f?1:0"%x)
'''lambda x:gpumap("$>=%f?1:0"%x)'''
gpunot   = gpumap("$<1.0f?1:0")
'''gpumap("$<1.0f?1:0")'''
gpuyes   = gpumap("$<1.0f?0:1")
'''gpumap("$<1.0f?0:1")'''
gpuand   = gpubinaryeq('($x+$y)>=2.0f?1.0f:0.0f')
'''gpubinaryeq('($x+$y)>=2.0f?1.0f:0.0f')'''
gpunor   = gpubinaryeq('($x+$y)<=0.0f?1.0f:0.0f')
'''gpubinaryeq('($x+$y)<=0.0f?1.0f:0.0f')'''
gpuhzero = lambda a,b:gpumap("$>=%f&&$<%d?$:0"%(a,b))
'''lambda a,b:gpumap("$>=%f&&$<%d?$:0"%(a,b))'''
gpurange = lambda a,b:gpumap("$>=%f&&$<%d?1:0"%(a,b))
'''lambda a,b:gpumap("$>=%f&&$<%d?1:0"%(a,b))'''


