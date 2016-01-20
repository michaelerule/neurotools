
# IF THIS FAILS THEN DO
# > sudo easy_install statsmodels
# AND RESTART PYTHON INTERPRETER

try: 
    import statsmodels.api as sm

    def glmfit(X,Y):
        '''
        
        THIS IS NOT A CLONE OF THE MATLAB GLMFIT. IT IS FOR POISSON PROCESS
        GLMS ONLY. IT DOES IMPLEMENT NEWTON-RAPHSON THOUGH.
        
        STATSMODELS HAS IMPLEMENTED ITERATIVELY REWEIGHTED LEAST SQUARES FOR US
        SO LETS TAKE ADVANTAGE OF THAT
        
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

        The statsmodels GLM code is listed as
        
        # Load modules and data
        import statsmodels.api as sm
        data = sm.datasets.scotland.load()
        data.exog = sm.add_constant(data.exog)

        # Instantiate a gamma family model with the default link function.
        gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
        gamma_results = gamma_model.fit()
        
        The docstring for statsmodels.api.GLM may be relevant. This is just a
        thin wrapper for that function.
        
            Generalized Linear Models class

        GLM inherits from statsmodels.base.model.LikelihoodModel

        Parameters
        -----------
        endog : array-like
            1d array of endogenous response variable.  This array can be 1d or 2d.
            Binomial family models accept a 2d array with two columns. If
            supplied, each observation is expected to be [success, failure].
        exog : array-like
            A nobs x k array where `nobs` is the number of observations and `k`
            is the number of regressors. An intercept is not included by default
            and should be added by the user (models specified using a formula
            include an intercept by default). See `statsmodels.tools.add_constant`.
        family : family class instance
            The default is Gaussian.  To specify the binomial distribution
            family = sm.family.Binomial()
            Each family can take a link instance as an argument.  See
            statsmodels.family.family for more information.
        missing : str
            Available options are 'none', 'drop', and 'raise'. If 'none', no nan
            checking is done. If 'drop', any observations with nans are dropped.
            If 'raise', an error is raised. Default is 'none.'

        Attributes
        -----------
        df_model : float
            `p` - 1, where `p` is the number of regressors including the intercept.
        df_resid : float
            The number of observation `n` minus the number of regressors `p`.
        endog : array
            See Parameters.
        exog : array
            See Parameters.
        family : family class instance
            A pointer to the distribution family of the model.
        mu : array
            The estimated mean response of the transformed variable.
        normalized_cov_params : array
            `p` x `p` normalized covariance of the design / exogenous data.
        pinv_wexog : array
            For GLM this is just the pseudo inverse of the original design.
        scale : float
            The estimate of the scale / dispersion.  Available after fit is called.
        scaletype : str
            The scaling used for fitting the model.  Available after fit is called.
        weights : array
            The value of the weights after the last iteration of fit.

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.scotland.load()
        >>> data.exog = sm.add_constant(data.exog)

        Instantiate a gamma family model with the default link function.

        >>> gamma_model = sm.GLM(data.endog, data.exog,
        ...                      family=sm.families.Gamma())

        >>> gamma_results = gamma_model.fit()
        >>> gamma_results.params
        array([-0.01776527,  0.00004962,  0.00203442, -0.00007181,  0.00011185,
               -0.00000015, -0.00051868, -0.00000243])
        >>> gamma_results.scale
        0.0035842831734919055
        >>> gamma_results.deviance
        0.087388516416999198
        >>> gamma_results.pearson_chi2
        0.086022796163805704
        >>> gamma_results.llf
        -83.017202161073527

        See also
        --------
        statsmodels.genmod.families.family
        :ref:`families`
        :ref:`links`

        Notes
        -----
        Only the following combinations make sense for family and link ::

                       + ident log logit probit cloglog pow opow nbinom loglog logc
          Gaussian     |   x    x                        x
          inv Gaussian |   x    x                        x
          binomial     |   x    x    x     x       x     x    x           x      x
          Poission     |   x    x                        x
          neg binomial |   x    x                        x          x
          gamma        |   x    x                        x

        Not all of these link functions are currently available.

        Endog and exog are references so that if the data they refer to are already
        arrays and these arrays are changed, endog and exog will change.


        **Attributes**

        df_model : float
            Model degrees of freedom is equal to p - 1, where p is the number
            of regressors.  Note that the intercept is not reported as a
            degree of freedom.
        df_resid : float
            Residual degrees of freedom is equal to the number of observation n
            minus the number of regressors p.
        endog : array
            See above.  Note that endog is a reference to the data so that if
            data is already an array and it is changed, then `endog` changes
            as well.
        exposure : array-like
            Include ln(exposure) in model with coefficient constrained to 1. Can
            only be used if the link is the logarithm function.
        exog : array
            See above.  Note that endog is a reference to the data so that if
            data is already an array and it is changed, then `endog` changes
            as well.
        iteration : int
            The number of iterations that fit has run.  Initialized at 0.
        family : family class instance
            The distribution family of the model. Can be any family in
            statsmodels.families.  Default is Gaussian.
        mu : array
            The mean response of the transformed variable.  `mu` is the value of
            the inverse of the link function at lin_pred, where lin_pred is the linear
            predicted value of the WLS fit of the transformed variable.  `mu` is
            only available after fit is called.  See
            statsmodels.families.family.fitted of the distribution family for more
            information.
        normalized_cov_params : array
            The p x p normalized covariance of the design / exogenous data.
            This is approximately equal to (X.T X)^(-1)
        offset : array-like
            Include offset in model with coefficient constrained to 1.
        pinv_wexog : array
            The pseudoinverse of the design / exogenous data array.  Note that
            GLM has no whiten method, so this is just the pseudo inverse of the
            design.
            The pseudoinverse is approximately equal to (X.T X)^(-1)X.T
        scale : float
            The estimate of the scale / dispersion of the model fit.  Only
            available after fit is called.  See GLM.fit and GLM.estimate_scale
            for more information.
        scaletype : str
            The scaling used for fitting the model.  This is only available after
            fit is called.  The default is None.  See GLM.fit for more information.
        weights : array
            The value of the weights after the last iteration of fit.  Only
            available after fit is called.  See statsmodels.families.family for
            the specific distribution weighting functions.
        '''

        # check for and maybe add constant value to X
        if not all(X[:,0]==X[0,0]):
            X = hstack([ ones((shape(X)[0],1),dtype=X.dtype), X])
        
        poisson_model   = sm.GLM(Y,X,family=sm.families.Poisson())
        poisson_results = poisson_model.fit()
        M = poisson_results.params
        return M

except:
    print 'statsmodels is not installed! no GLMfit'
    print 'TRY'
    print ' > sudo easy_install statsmodels'
    print 'AND RESTART PYTHON INTERPRETER'
    
    


