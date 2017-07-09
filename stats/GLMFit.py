#!/usr/bin/python
# -*- coding: UTF-8 -*-
# PYTHON 2/3 COMPATIBILITY BOILERPLATE
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function


try:
    import statsmodels.api as sm

    def glmfit(X,Y):
        '''
        Wrapper for statsmodels glmfit that prepares a constant 
        parameter and configuration options for poisson-GLM fitting.
        Please see the documentation for glmfit in statsmodels for
        more details. 
        
        This method will automatically add a constant colum to the feature
        matrix Y

        Parameters
        -----------
        X : array-like
            A nobs x k array where `nobs` is the number of observations and `k`
            is the number of regressors. An intercept is not included by default
            and should be added by the user (models specified using a formula
            include an intercept by default). See `statsmodels.tools.add_constant`.
        Y : array-like
            1d array of poisson counts.  This array can be 1d or 2d.
        '''
        # check for and maybe add constant value to X
        if not all(X[:,0]==X[0,0]):
            X = hstack([ ones((shape(X)[0],1),dtype=X.dtype), X])

        poisson_model   = sm.GLM(Y,X,family=sm.families.Poisson())
        poisson_results = poisson_model.fit()
        M = poisson_results.params
        return M

except:
    print('statsmodels is not installed! no GLMfit')
    print('TRY')
    print(' > sudo easy_install statsmodels')
    print('AND RESTART PYTHON INTERPRETER')
