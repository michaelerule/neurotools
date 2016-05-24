#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
You know what would be really cool? If we combined advanced caching with
lazy evaluation. Haskell style. How might we do this? 

We cannot know the value we need in advance, so we need to actually step
through the code to find it out. What we need is a decorator that returns
a "thunk" which is an un-evaluated bit of code that we can use to grab
the result later. Is there a transparent way to do this? 

If you allow thunks to propagate, for example, by consuming numerical 
expressions via operator overloading, a whole world of hell breaks loose.
I think this can be implemented but on no accounts SHOULD it be implemented.

Or maybe, maybe it should be?
SEE
https://pypi.python.org/pypi/lazy_python



'''
